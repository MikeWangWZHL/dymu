""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in:

'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929

`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270

`FlexiViT: One Model for All Patch Sizes`
    - https://arxiv.org/abs/2212.08013

The official jax code is released and available at
  * https://github.com/google-research/vision_transformer
  * https://github.com/google-research/big_vision

Acknowledgments:
  * The paper authors for releasing code and weights, thanks!
  * I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch
  * Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
  * Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020, Ross Wightman
"""
import copy
import logging
import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
from einops import rearrange
import ipdb

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, \
    OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.layers import PatchEmbed, Mlp, DropPath, AttentionPoolLatent, RmsNorm, PatchDropout, SwiGLUPacked, SwiGLU, \
    trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, use_fused_attn, \
    get_act_layer, get_norm_layer, LayerType
from timm.models._builder import build_model_with_cfg
from timm.models._features import feature_take_indices
from timm.models._manipulate import named_apply, checkpoint_seq, adapt_input_conv
from timm.models._registry import generate_default_cfgs, register_model, register_model_deprecations

from timm.models.vision_transformer import Attention, Block, LayerScale, VisionTransformer, global_pool_nlc
from timm.models.vision_transformer import _create_vision_transformer
__all__ = ['VisionTransformer']  # model_registry will add each entrypoint fn to this


_logger = logging.getLogger(__name__)


## ToME utils



def do_nothing(x, mode=None):
    return x


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)
        

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")
    x = x / size
    return x, size


def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source


#####################
def batch_level_bipartite_soft_matching(
    metric: torch.Tensor, # (b, s, c)
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
    padding_mask: Optional[torch.Tensor] = None, # (b, s) # 0 for non padding, 1 for padding
    max_r_per_instance: int = None,
    specified_threshold: float = None
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    bsz, seq_len, hdim = metric.shape

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        # compute scores within instance
        metric = metric / metric.norm(dim=-1, keepdim=True) # (b, s, c)
        # print(metric)

        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2) # (b, s//2, s//2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        # add padding mask
        if padding_mask is not None:
            # Create padding masks for 'a' and 'b'
            padding_mask_a = padding_mask[..., ::2]  # Shape: (b, s//2)
            padding_mask_b = padding_mask[..., 1::2]  # Shape: (b, s//2)

            # Unsqueeze to align dimensions for broadcasting
            mask_a = padding_mask_a.unsqueeze(2).bool()  # Shape: (b, s//2, 1)
            mask_b = padding_mask_b.unsqueeze(1).bool()  # Shape: (b, 1, s//2)

            # Combine masks to identify where either 'a' or 'b' has padding
            combined_mask = mask_a | mask_b  # Shape: (b, s//2, s//2)

            # Set scores at padding positions to -inf
            scores = scores.masked_fill(combined_mask, -math.inf)

        if max_r_per_instance is not None:
            node_max_instance, node_idx_instance = scores.max(dim=-1)
            edge_idx_instance = node_max_instance.argsort(dim=-1, descending=True)[..., None]
            unm_idx_instance = edge_idx_instance[..., max_r_per_instance:, :] # keep tokens beyond r_max unmerged
            unm_idx_instance_expanded = unm_idx_instance.expand(-1, -1, scores.size(-1))
            batch_indices = torch.arange(bsz).view(-1, 1, 1).expand_as(unm_idx_instance_expanded)
            scores[batch_indices, unm_idx_instance_expanded, :] = -math.inf

        # flatten across batch
        scores = rearrange(scores, 'b i j -> (b i) j')

        # get the best matching over the batch
        node_max, node_idx = scores.max(dim=-1) # (b * s // 2)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        if specified_threshold is not None:
            rb = sum(node_max > specified_threshold) # merge all tokens over the specified_threshold
        else:
            rb = r * bsz

        unm_idx = edge_idx[rb:, :]  # Unmerged Tokens (unmerged_token_num, 1)
        src_idx = edge_idx[:rb, :]  # Merged Tokens (rb, 1)
        dst_idx = node_idx.gather(dim=0, index=src_idx.squeeze()) # (rb,)

        if specified_threshold is not None:
            batch_threshold = None
        else:
            # keep track of batch level threshold for this layer
            batch_threshold = node_max[edge_idx[rb, 0]]
            batch_threshold = max(batch_threshold, torch.zeros_like(batch_threshold)) # should be non-negative
        # print(scores)
        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def update_dst(dst_flat, src_elements, index_expanded, reduce='sum', include_self=True):
        if reduce == 'sum':
            # Use scatter_add_ for sum reduction
            dst_flat.scatter_add_(
                dim=0,
                index=index_expanded,
                src=src_elements
            )
        elif reduce == 'mean':
            # For mean reduction, we'll need to keep track of counts
            counts = torch.zeros_like(dst_flat)
            ones = torch.ones_like(src_elements)

            # Sum the src_elements into dst_flat
            sum_dst_flat = torch.zeros_like(dst_flat)
            sum_dst_flat.scatter_add_(
                dim=0,
                index=index_expanded,
                src=src_elements
            )

            # Count the number of times each index is updated
            counts.scatter_add_(
                dim=0,
                index=index_expanded,
                src=ones
            )

            if include_self:
                # Include original dst values in the mean calculation
                sum_dst_flat += dst_flat
                counts += (dst_flat != 0).float()

            # Avoid division by zero
            counts = counts.clamp(min=1)

            # Compute the mean
            dst_flat = sum_dst_flat / counts
        else:
            raise ValueError("Unsupported reduction type. Use 'sum' or 'mean'.")

        return dst_flat


    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]  # Shape: (b, s//2, c)

        # src = rearrange(src, 'b s c -> (b s) c') # (b * s // 2, c)
        t1 = src.shape[1]
        unm_b = unm_idx.squeeze(-1) // t1
        unm_s = unm_idx.squeeze(-1) % t1
        src_b = src_idx.squeeze(-1) // t1
        src_s = src_idx.squeeze(-1) % t1

        # print("unm_idx", unm_idx)
        # print("src_idx", src_idx)
        # print("unm_b", unm_b)
        # print("unm_s", unm_s)
        # print("src_b", src_b)
        # print("src_s", src_s)
        # print("dst_idx", dst_idx)

        # fill in unmerged src tokens
        # src_new[unm_b, unm_s, :] = src[unm_b, unm_s, :]

        # # fill in merged tokens to dst locations
        # src_tokens = src[src_b, src_s, :]  # Shape: (b * r, c)
        # dst_tokens = dst[src_b, dst_idx, :]  # Shape: (b * r, c)
        # if mode == "mean":
        #     merged_tokens = (src_tokens + dst_tokens) / 2
        # elif mode == "sum":
        #     merged_tokens = src_tokens + dst_tokens
        # else:
        #     raise ValueError(f"Unsupported merge mode: {mode}")
        # dst_new[src_b, dst_idx, :] = merged_tokens

        src_tokens = src[src_b, src_s, :]  # Shape: (b * r, c)
        dst_b = src_b
        dst_seq_len = dst.size(1)
        dst_flat_indices = dst_b * dst_seq_len + dst_idx
        dst_flat = dst.reshape(-1, dst.size(-1))
        index_expanded = dst_flat_indices.unsqueeze(-1).expand(-1, dst_flat.size(-1))
        dst_new = dst_flat.clone()
        dst_new = update_dst(dst_new, src_tokens, index_expanded, reduce=mode, include_self=True)
        dst_new = dst_new.reshape(dst.size())
        # print("dst tokens merged:", dst_new)

        # # construct new x
        x_new = x.clone()
        x_new[..., :src.size(1), :] = src
        x_new[..., src.size(1):, :] = dst_new

        if padding_mask is not None:
            padding_mask_src, padding_mask_dst = padding_mask[..., ::2].clone(), padding_mask[..., 1::2].clone()
            padding_mask_src[src_b, src_s] = 1
            new_padding_mask = torch.concatenate((padding_mask_src, padding_mask_dst), dim=1)
        else:
            new_padding_mask = torch.zeros((bsz, seq_len), device=x.device, dtype=x.dtype)
            new_padding_mask_a = new_padding_mask[..., ::2].clone()
            new_padding_mask_a[src_b, src_s] = 1
            new_padding_mask[..., :src.size(1)] = new_padding_mask_a
        
        # # construct padding masking: (b, s); 0 for non-padding, 1 for padding; fill in 1 where x_new is zero
        # padding_mask = torch.all(x_new == 0, dim=-1).int().to(x.device)
        return x_new, new_padding_mask


    def unmerge(x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement unmerge
        raise NotImplementedError("Unmerge not implemented yet.")
        # unm_len = unm_idx.shape[1]
        # unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        # n, _, c = unm.shape

        # src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        # out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        # out[..., 1::2, :] = dst
        # out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        # out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)
        return out

    return merge, unmerge, batch_threshold

def batch_level_merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x, padding_mask = merge(x * size, mode="sum")
    size, _ = merge(size, mode="sum")

    if padding_mask is not None:
        # Rearrange x, padding_mask, and size so that non-padding instances are at the front
        # padding_mask is 0 for non-padding and 1 for padding
        sort_indices = torch.argsort(padding_mask, dim=1)
        # Use gather to rearrange x, padding_mask, and size according to sort_indices
        x = x.gather(1, sort_indices.unsqueeze(-1).expand(-1, -1, x.size(2)))
        padding_mask = padding_mask.gather(1, sort_indices)
        size = size.gather(1, sort_indices.unsqueeze(-1).expand(-1, -1, size.size(2)))

    x = x / size

    # Truncate to the maximum length
    max_len = int((1 - padding_mask).sum(dim=-1).max().item())
    x = x[:, :max_len]
    padding_mask = padding_mask[:, :max_len]
    size = size[:, :max_len]
    return x, size, padding_mask

def batch_level_merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    raise NotImplementedError("Unmerge not implemented yet.")



##


class ToMEAttention(Attention):


    def forward(self, x: torch.Tensor, size: Optional[torch.Tensor] = None, # ToMe token size vector
                ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn and False:
            raise NotImplementedError
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            ## apply ToMe proportional attention here
            if size is not None:
                size_bias_log = size.log()[:, :, 0] # (b, src_len, 1) -> (b, src_len)
                size_bias_log = size_bias_log.unsqueeze(1).unsqueeze(1).repeat(1, self.num_heads, 1, 1) # (b, src_len) -> (b, num_heads, 1, src_len)
                attn = attn + size_bias_log
                # bs*nhead, Nk, d
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        ## ToMe cosine similarity metric 
        metric = k.view(B, self.num_heads, N, self.head_dim).mean(1)
        return x, metric




class ToMEBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            mlp_layer: Type[nn.Module] = Mlp,
            # TOME args
            trace_source: bool = False,
            prop_attn: bool = True,
            cls_token: bool = True,
            r: int = 0,
            merge_mode: str = "instance_level",
            max_r_per_instance_ratio: float = None,
            update_threshold: bool = False,
            specified_threshold: float = None
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ToMEAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            bias=proj_bias,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # ToMe configs
        self._tome_info = {
            "r": r, # number of tokens to remove
            "size": None,
            "source": None,
            "trace_source": trace_source,
            "prop_attn": prop_attn,
            "class_token": cls_token,
            "distill_token": False,
            "merge_mode": merge_mode,
            "max_r_per_instance_ratio": max_r_per_instance_ratio,
        }
        if max_r_per_instance_ratio is not None:
            print("setting max r per instance to: ", int(self._tome_info["max_r_per_instance_ratio"] * r))

        self.update_threshold = update_threshold
        if r>0:
            # self.register_buffer('threshold', torch.tensor(1.0)) # default to be no merging
            self.threshold = torch.tensor(1.0)
        self.momentum = 0.1
        self.specified_threshold = specified_threshold


    def threshold_running_avg(self, new_value):
        if new_value is not None:
            with torch.no_grad():
                if torch.all(self.threshold == 1.0):
                    self.threshold = new_value
                else:
                    if new_value.device != self.threshold.device:
                        new_value = new_value.to(self.threshold.device)
                    self.threshold = (1-self.momentum) * self.threshold + self.momentum * new_value
        
    def merge_tokens(self, metric, r, hidden_states, padding_mask=None):
        if self._tome_info["merge_mode"] == "instance_level":
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, hidden_states, self._tome_info["source"]
                )
            hidden_states, self._tome_info["size"] = merge_wavg(
                merge, hidden_states, self._tome_info["size"]
            )
        elif self._tome_info["merge_mode"] == "batch_level":

            if not self.training and not self.update_threshold:
                if self.specified_threshold is not None:
                    specified_threshold = self.specified_threshold
                else:
                    specified_threshold = self.threshold
                # print("using specified threshold: ", specified_threshold)
            else:
                specified_threshold = None

            if self._tome_info["max_r_per_instance_ratio"] is None:
                max_r_per_instance = None
            else:
                max_r_per_instance = int(self._tome_info["max_r_per_instance_ratio"] * r)
            merge, _, batch_threshold = batch_level_bipartite_soft_matching(
                metric,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
                padding_mask = padding_mask,
                max_r_per_instance = max_r_per_instance,
                specified_threshold = specified_threshold
            )
            hidden_states, self._tome_info["size"], padding_mask = batch_level_merge_wavg(
                merge, hidden_states, self._tome_info["size"]
            )
            if self.training or self.update_threshold:
                self.threshold_running_avg(batch_threshold)
        return hidden_states, padding_mask




    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        x, metric = self.attn(x, size=attn_size)
        x = self.drop_path1(x)
        x = x + residual
        # x = x + self.drop_path1(self.ls1(self.attn())) 
        r = self._tome_info["r"]
        if r > 0:
            x, padding_mask = self.merge_tokens(metric, r, x)
        
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x



class ToMEVisionTransformer(VisionTransformer):

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def pool(self, x: torch.Tensor, pool_type: Optional[str] = None) -> torch.Tensor:
        if self.attn_pool is not None:
            x = self.attn_pool(x)
            return x
        pool_type = self.global_pool if pool_type is None else pool_type
        x = global_pool_nlc(x, pool_type=pool_type, num_prefix_tokens=self.num_prefix_tokens)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.pool(x)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x



@register_model
def vit_base_patch16_siglip_224_tome(pretrained: bool = False, **kwargs) -> VisionTransformer:
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, class_token=False, global_pool='map',
    )
    model = _create_vision_transformer(
        'vit_base_patch16_siglip_224', pretrained=pretrained, block_fn=partial(ToMEBlock, merge_mode="batch_level", r=32), **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch16_siglip_256_tome(pretrained: bool = False, **kwargs) -> VisionTransformer:
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, class_token=False, global_pool='map',
    )
    model = _create_vision_transformer(
        'vit_base_patch16_siglip_256', pretrained=pretrained, block_fn=partial(ToMEBlock, merge_mode="batch_level", r=32), **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch16_siglip_384_tome(pretrained: bool = False, **kwargs) -> VisionTransformer:
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, class_token=False, global_pool='map',
    )
    model = _create_vision_transformer(
        'vit_base_patch16_siglip_384', pretrained=pretrained, block_fn=partial(ToMEBlock, merge_mode="batch_level", r=32), **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch16_siglip_512_tome(pretrained: bool = False, **kwargs) -> VisionTransformer:
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, class_token=False, global_pool='map',
    )
    model = _create_vision_transformer(
        'vit_base_patch16_siglip_512', pretrained=pretrained, block_fn=partial(ToMEBlock, merge_mode="batch_level", r=32), **dict(model_args, **kwargs))
    return model


@register_model
def vit_large_patch16_siglip_256_tome(pretrained: bool = False, **kwargs) -> VisionTransformer:
    model_args = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, class_token=False, global_pool='map',
    )
    model = _create_vision_transformer(
        'vit_large_patch16_siglip_256', pretrained=pretrained, block_fn=partial(ToMEBlock, merge_mode="batch_level", r=32), **dict(model_args, **kwargs))
    return model


@register_model
def vit_large_patch16_siglip_384_tome(pretrained: bool = False, **kwargs) -> VisionTransformer:
    model_args = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, class_token=False, global_pool='map',
    )
    model = _create_vision_transformer(
        'vit_large_patch16_siglip_384', pretrained=pretrained, block_fn=partial(ToMEBlock, merge_mode="batch_level", r=32), **dict(model_args, **kwargs))
    return model


@register_model
def vit_so400m_patch14_siglip_224_tome(pretrained: bool = False, **kwargs) -> VisionTransformer:
    model_args = dict(
        patch_size=14, embed_dim=1152, depth=27, num_heads=16, mlp_ratio=3.7362, class_token=False, global_pool='map',
    )
    model = _create_vision_transformer(
        'vit_so400m_patch14_siglip_224', pretrained=pretrained, block_fn=partial(ToMEBlock, merge_mode="batch_level", r=32), **dict(model_args, **kwargs))
    return model


@register_model
def vit_so400m_patch16_siglip_256_tome(pretrained: bool = False, **kwargs) -> VisionTransformer:
    # this is a corrected variant of the 384 with a res properly divisible by patch size (no padding/truncation)
    model_args = dict(
        patch_size=16, embed_dim=1152, depth=27, num_heads=16, mlp_ratio=3.7362, class_token=False, global_pool='map',
    )
    model = _create_vision_transformer(
        'vit_so400m_patch16_siglip_256', pretrained=pretrained, block_fn=partial(ToMEBlock, merge_mode="batch_level", r=32), **dict(model_args, **kwargs))
    return model


@register_model
def vit_so400m_patch14_siglip_378_tome(pretrained: bool = False, **kwargs) -> VisionTransformer:
    # this is a corrected variant of the 384 with a res properly divisible by patch size (no padding/truncation)
    model_args = dict(
        patch_size=14, embed_dim=1152, depth=27, num_heads=16, mlp_ratio=3.7362, class_token=False, global_pool='map',
    )
    model = _create_vision_transformer(
        'vit_so400m_patch14_siglip_378', pretrained=pretrained, block_fn=partial(ToMEBlock, merge_mode="batch_level", r=32), **dict(model_args, **kwargs))
    return model


@register_model
def vit_so400m_patch14_siglip_384_tome(pretrained: bool = False, **kwargs) -> VisionTransformer:
    model_args = dict(
        patch_size=14, embed_dim=1152, depth=27, num_heads=16, mlp_ratio=3.7362, class_token=False, global_pool='map',
    )
    model = _create_vision_transformer(
        'vit_so400m_patch14_siglip_384', pretrained=pretrained, block_fn=partial(ToMEBlock, merge_mode="batch_level", r=32), **dict(model_args, **kwargs))
    return model

