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

import ipdb
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from functools import partial
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
from einops import rearrange


from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, \
    OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.layers import PatchEmbed, Mlp, DropPath, AttentionPoolLatent, RmsNorm, PatchDropout, SwiGLUPacked, SwiGLU, \
    trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, use_fused_attn, \
    get_act_layer, get_norm_layer, LayerType
from timm.models._builder import build_model_with_cfg
from timm.models._features import feature_take_indices
from timm.models._manipulate import named_apply, checkpoint_seq, adapt_input_conv, checkpoint
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
        return do_nothing, do_nothing, None

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
                # counts += (dst_flat != 0).float()
                counts += (dst_flat != 0).to(counts.dtype)

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
    # max_len = int((1 - padding_mask).sum(dim=-1).max().item()) # this causes incorrect max_len calculation
    max_len = int((1 - padding_mask).to(torch.int64).sum(dim=-1).max().item())
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
                attention_mask: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # (3, B, num_heads, N, head_dim)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k) # (B, num_heads, N, head_dim)

        full_bias = None
        if size is not None:
            size_bias_log = size.log()[:, :, 0] # (b, src_len, 1) -> (b, src_len)
            size_bias_log = size_bias_log.unsqueeze(1).unsqueeze(1).expand(B, self.num_heads, N, N) # (b, src_len) -> (b, num_heads, 1, src_len)
            full_bias = size_bias_log
        
        # apply attention mask before softmax (-inf for masked tokens)
        if attention_mask is not None:
            if attention_mask.size() != (B, 1, N, N):
                raise ValueError(
                    f"Attention mask should be of size {(B, 1, N, N)}, but is {attention_mask.size()}"
                )
            if full_bias is None:
                full_bias = 0
            full_bias = full_bias + attention_mask
            
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=full_bias
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1) # (B, num_heads, N, N)
            ## apply ToMe proportional attention here
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            if full_bias is not None:
                attn = attn + full_bias
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
        # if r>0:
        self.register_buffer('threshold', torch.tensor(1.0)) # default to be no merging
        # self.threshold = torch.tensor(1.0)
        self.threshold_count = 1.0
        self.specified_threshold = specified_threshold

    def threshold_running_avg(self, new_value):
        if new_value is not None:
            with torch.no_grad():
                if torch.all(self.threshold == 1.0):
                    self.threshold = new_value
                else:
                    if new_value.device != self.threshold.device:
                        new_value = new_value.to(self.threshold.device)
                    # self.threshold = (1-self.momentum) * self.threshold + self.momentum * new_value
                    self.threshold = (self.threshold*self.threshold_count + new_value)/(self.threshold_count+1)
                    self.threshold_count+=1
                    # print(f'New threshold: {self.threshold}')
                    if dist.is_initialized() and dist.get_world_size() > 1:
                        dist.all_reduce(self.threshold, op=dist.ReduceOp.AVG)

        
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
            if merge != do_nothing:
                hidden_states, self._tome_info["size"], padding_mask = batch_level_merge_wavg(
                    merge, hidden_states, self._tome_info["size"]
                )
                if self.training or self.update_threshold:
                    self.threshold_running_avg(batch_threshold)
            
        return hidden_states, padding_mask

    def _get_attn_mask_from_padding_mask(self, padding_mask, dtype):
        """
            input: padding mask: (b, s): 0 for non-padding, 1 for padding
            output: attention mask: (b, 1, s, s): 0 for non-padding, -inf for padding
        """
        # Expand padding mask to match attention mask shape
        attn_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # Shape: (b, 1, 1, s)
        attn_mask = attn_mask.expand(-1, 1, padding_mask.size(1), -1)  # Shape: (b, 1, s, s)
        # Convert padding positions to -inf
        attn_mask = attn_mask * torch.finfo(dtype).min
        return attn_mask

    def forward(self, 
        x: torch.Tensor,  
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.norm1(x)

        dtype = x.dtype
        if padding_mask is not None:
            attention_mask = self._get_attn_mask_from_padding_mask(padding_mask, dtype)
        else:
            attention_mask = None

        ## ToMe proportional attention
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        
        x, metric = self.attn(x, size=attn_size, attention_mask=attention_mask)
        x = self.drop_path1(x)
        x = x + residual
        # x = x + self.drop_path1(self.ls1(self.attn())) 
        r = self._tome_info["r"]
        if r > 0:
            x, padding_mask = self.merge_tokens(metric, r, x, padding_mask=padding_mask)
        
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        outputs = {
            "hidden_states": x,
            "padding_mask": padding_mask
        }
        return outputs

class AttentionPoolLatentWMasking(AttentionPoolLatent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Ensure parent initialization

        # Ensure `self.latent` has correct shape (at least [1, 1, embed_dim])
        if self.latent_len == 0:
            self.latent = nn.Parameter(torch.randn(1, 1, self.latent_dim))  # Ensure non-empty tensor
        print("attn_pool: self.latent.shape:", self.latent.shape)

    def forward(self, x, 
                attention_mask=None, 
                size=None # ToMe token size vector
        ):
        B, N, C = x.shape

        if self.pos_embed is not None:
            # FIXME interpolate
            x = x + self.pos_embed.unsqueeze(0).to(x.dtype)

        q_latent = self.latent.expand(B, -1, -1)
        q = self.q(q_latent).reshape(B, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)

        full_bias = None
        if size is not None:
            size_bias_log = size.log()[:, :, 0] # (b, src_len, 1) -> (b, src_len)
            size_bias_log = size_bias_log.unsqueeze(1).unsqueeze(1).expand(B, self.num_heads, 1, N) # (b, src_len) -> (b, num_heads, 1, src_len)
            full_bias = size_bias_log

        if attention_mask is not None:
            assert attention_mask.size() == (B, 1, self.latent_len, N) or attention_mask.size() == (B, 1, 1, N), f"Attention mask shape {attention_mask.size()} not compatible with input shape {B, 1, self.latent_len, N}"
            if full_bias is None:
                full_bias = 0
            full_bias = full_bias + attention_mask
        
        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=full_bias)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1) # (B, num_heads, N, N)
            if full_bias is not None:
                attn = attn + full_bias
            attn = attn.softmax(dim=-1)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, self.latent_len, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x + self.mlp(self.norm(x))

        # optional pool if latent seq_len > 1 and pooled output is desired
        if self.pool == 'token':
            x = x[:, 0]
        elif self.pool == 'avg':
            x = x.mean(1)
        return x

def global_pool_nlc_w_masking(
    x: torch.Tensor,
    pool_type: str = 'token',
    num_prefix_tokens: int = 1,
    reduce_include_prefix: bool = False,
    padding_mask: Optional[torch.Tensor] = None,  # (B, S) with 0 for valid, 1 for padding
):
    # If no pooling is requested, just return x
    if not pool_type:
        return x

    if pool_type == 'token':
        # For token pooling, we simply return the first token (assumed to be a [CLS] token)
        x = x[:, 0]
    else:
        # Optionally exclude the prefix tokens from pooling
        if not reduce_include_prefix:
            x = x[:, num_prefix_tokens:]
            if padding_mask is not None:
                padding_mask = padding_mask[:, num_prefix_tokens:]
                
        # Average Pooling: sum valid tokens and divide by the count of valid tokens
        if pool_type == 'avg':
            if padding_mask is None:
                x = x.mean(dim=1)
            else:
                # Create a mask of valid tokens (shape: B x S x 1)
                valid = (padding_mask == 0).unsqueeze(-1).to(x.dtype)
                # Sum only over valid tokens
                x_sum = (x * valid).sum(dim=1)
                # Compute the number of valid tokens per example, avoid division by zero
                valid_count = valid.sum(dim=1).clamp(min=1)
                x = x_sum / valid_count

        # Max Pooling: ignore padded tokens by setting them to -inf
        elif pool_type == 'max':
            if padding_mask is None:
                x = x.amax(dim=1)
            else:
                # Create mask for padded tokens (True where padded)
                mask = (padding_mask == 1).unsqueeze(-1)
                # Replace padded tokens with -inf so they don't affect the max
                x_masked = x.masked_fill(mask, float('-inf'))
                x = x_masked.amax(dim=1)

        # AvgMax Pooling: average of average and max pooling
        elif pool_type == 'avgmax':
            if padding_mask is None:
                avg = x.mean(dim=1)
                max_val = x.amax(dim=1)
            else:
                valid = (padding_mask == 0).unsqueeze(-1).to(x.dtype)
                avg = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
                mask = (padding_mask == 1).unsqueeze(-1)
                x_masked = x.masked_fill(mask, float('-inf'))
                max_val = x_masked.amax(dim=1)
            x = 0.5 * (avg + max_val)
        else:
            raise ValueError(f'Unknown pool type {pool_type}')

    return x


from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
from typing import Optional, Tuple

@dataclass
class CLIPVisionEncoderToMEOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    sizes: Optional[Tuple[torch.FloatTensor, ...]] = None
    padding_masks: Optional[Tuple[torch.FloatTensor, ...]] = None

class ToMEVisionTransformer(VisionTransformer):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """
    dynamic_img_size: Final[bool]

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: Literal['', 'avg', 'avgmax', 'max', 'token', 'map'] = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            proj_bias: bool = True,
            init_values: Optional[float] = None,
            class_token: bool = True,
            pos_embed: str = 'learn',
            no_embed_class: bool = False,
            reg_tokens: int = 0,
            pre_norm: bool = False,
            final_norm: bool = True,
            fc_norm: Optional[bool] = None,
            dynamic_img_size: bool = False,
            dynamic_img_pad: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: Literal['skip', 'jax', 'jax_nlhb', 'moco', ''] = '',
            fix_init: bool = False,
            embed_layer: Callable = PatchEmbed,
            embed_norm_layer: Optional[LayerType] = None,
            norm_layer: Optional[LayerType] = None,
            act_layer: Optional[LayerType] = None,
            block_fn: Type[nn.Module] = Block,
            mlp_layer: Type[nn.Module] = Mlp,
            # tome args
            merge_mode: str = "batch_level", # merge mode: instance_level or batch_level
            r_total: int = 0, # total number of tokens to remove
            r_schedule: str = "constant", # r schedule: constant, linear, reverse_linear
            max_r_per_instance_ratio: float = None, # 1.0 => refer to fixed r for each instance; > 1.0 => dynamic r
            update_threshold: bool = False, # whether to post-hoc update threshold after training
            specified_thresholds: List[float] = None, # specified threshold for each layer
            **kwargs
    ) -> None:
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Number of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            no_embed_class: Don't include position embeddings for class (or reg) tokens.
            reg_tokens: Number of register tokens.
            pre_norm: Enable norm after embeddings, before transformer blocks (standard in CLIP ViT).
            final_norm: Enable norm after transformer blocks, before head (standard in most ViT).
            fc_norm: Move final norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            fix_init: Apply weight initialization fix (scaling w/ layer index).
            embed_layer: Patch embedding layer.
            embed_norm_layer: Normalization layer to use / override in patch embed module.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'avgmax', 'max', 'token', 'map')
        assert class_token or global_pool != 'token'
        assert pos_embed in ('', 'none', 'learn')
        use_fc_norm = global_pool in ('avg', 'avgmax', 'max') if fc_norm is None else fc_norm
        norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        embed_norm_layer = get_norm_layer(embed_norm_layer)
        act_layer = get_act_layer(act_layer) or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim  # for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class  # don't embed prefix positions (includes reg)
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False

        ### ToMe configs ###
        self.rs = self._get_rs(depth, r_total, r_schedule)
        print("set total avg remove token nums each layer as: ", self.rs)
        print("merge mode: ", merge_mode)
        self._tome_info = {
            "size": None,
            "source": None,
            "trace_source": False,
            "prop_attn": True,
            "class_token": self.has_class_token,
            "distill_token": False,
            "merge_mode": merge_mode,
            "max_r_per_instance_ratio": max_r_per_instance_ratio,
            "update_threshold": update_threshold,
            "specified_thresholds": specified_thresholds,
        }
        ###

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        if embed_norm_layer is not None:
            embed_args['norm_layer'] = embed_norm_layer
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches
        reduction = self.patch_embed.feat_ratio() if hasattr(self.patch_embed, 'feat_ratio') else patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        if not pos_embed or pos_embed == 'none':
            self.pos_embed = None
        else:
            self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        assert block_fn == ToMEBlock
        self.blocks =nn.ModuleList([
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                proj_bias=proj_bias,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
                ## ToMe args
                trace_source = False,
                prop_attn = True,
                cls_token = self.has_class_token,
                r = self.rs[i],
                merge_mode = merge_mode,
                max_r_per_instance_ratio = max_r_per_instance_ratio,
                update_threshold = update_threshold,
                specified_threshold = specified_thresholds[i] if specified_thresholds is not None else None
            )
            for i in range(depth)])
        
        self.feature_info = [
            dict(module=f'blocks.{i}', num_chs=embed_dim, reduction=reduction) for i in range(depth)]
        self.norm = norm_layer(embed_dim) if final_norm and not use_fc_norm else nn.Identity()

        # Classifier Head
        if global_pool == 'map':
            self.attn_pool = AttentionPoolLatentWMasking(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
            )
        else:
            self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if final_norm and use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)
        if fix_init:
            self.fix_init_weight()


    def get_layer_thresholds(self):
        return [layer.threshold for layer in self.blocks]
    
    def set_layer_thresholds(self, thresholds):
        dtype = self.blocks[0].threshold.dtype
        device = self.blocks[0].threshold.device
        assert len(thresholds) == len(self.blocks)
        for layer, th in zip(self.blocks, thresholds):
            if isinstance(th, torch.Tensor):
                layer.threshold = th.to(dtype=dtype, device=device)
            else:
                layer.threshold = torch.tensor(th, dtype=dtype, device=device)

    def _get_rs(self, num_layers, r_total, r_schedule="constant"):
        
        if r_total == 0:
            return [0] * num_layers
        
        if r_schedule == "constant":
            if r_total % num_layers == 0:
                r = r_total // num_layers
                return [r] * num_layers
            else:
                # Distribute as evenly as possible, but account for remainders
                base_r = r_total // num_layers
                remainder = r_total % num_layers
                # Create a distribution list starting with the base value
                distribution = [base_r] * num_layers
                # Distribute the remainder across the first few layers
                for i in range(remainder):
                    distribution[i] += 1
                return distribution

        elif r_schedule in ["linear", "reverse_linear"]:
            # approximate a linear schedule with the last layer has no reduction
            M = r_total
            N = num_layers
            r0 = (2*M) // N
            step = r0 / N
            s = []
            while sum(s) + int(r0 - len(s)*step) < M:
                s.append(int(r0 - len(s)*step))
            if sum(s) < M:
                s.append(M - sum(s))
            while len(s) < N:
                s.append(0)
            assert sum(s) == M
            assert len(s) == N
            if r_schedule == "linear":
                return s
            else:
                return s[::-1]
        else:
            raise ValueError(f"Invalid r_schedule: {r_schedule}")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        self._tome_info["size"] = None
        self._tome_info["source"] = None

        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        # if self.grad_checkpointing and not torch.jit.is_scripting():
        #     x = checkpoint_seq(self.blocks, x)
        # else:
        #     x = self.blocks(x)
        
        # using self.blocks as a nn.ModuleList
        self.output_stats = {}
        padding_mask = None
        for idx, block in enumerate(self.blocks):
            block._tome_info["size"] = self._tome_info["size"]
            if self.grad_checkpointing and not torch.jit.is_scripting():
                outputs = checkpoint(block, x, padding_mask)
            else:
                outputs = block(x, padding_mask=padding_mask)
            x, padding_mask = outputs["hidden_states"], outputs["padding_mask"]
            self._tome_info["size"] = block._tome_info["size"]
            if padding_mask is not None:
                ntoks = (padding_mask<0.5).float().sum(-1)
                ntoks = ntoks.detach().tolist()
            else:
                ntoks = [x.shape[1]]*x.shape[0]
            self.output_stats[f"block_{idx}_ntoks"] = ntoks
        final_size = self._tome_info["size"]
        x = self.norm(x)
        return x, padding_mask, final_size

    def forward_features_all_layers(self, x: torch.Tensor # (B, C, H, W)
                                    ) -> torch.Tensor:
        # for enable getting intermediate features for llava-style model
        self._tome_info["size"] = None
        self._tome_info["source"] = None

        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        
        # using self.blocks as a nn.ModuleList
        hidden_states = []
        padding_masks = []
        sizes = []
        padding_mask = None
        self.output_stats = {}
        for idx, block in enumerate(self.blocks):
            block._tome_info["size"] = self._tome_info["size"]
            if self.grad_checkpointing and not torch.jit.is_scripting():
                outputs = checkpoint(block, x, padding_mask)
            else:
                outputs = block(x, padding_mask=padding_mask)
            x, padding_mask = outputs["hidden_states"], outputs["padding_mask"]
            self._tome_info["size"] = block._tome_info["size"]
            hidden_states.append(x)
            padding_masks.append(padding_mask)
            sizes.append(self._tome_info["size"])
            # track number of tokens after each block
            if padding_mask is not None:
                ntoks = (padding_mask<0.5).float().sum(-1)
                ntoks = ntoks.detach().tolist()
            else:
                ntoks = [x.shape[1]]*x.shape[0]
            self.output_stats[f"block_{idx}_ntoks"] = ntoks
        
        x = self.norm(x)
        return CLIPVisionEncoderToMEOutput(
            last_hidden_state=x,
            hidden_states=tuple(hidden_states),
            padding_masks=tuple(padding_masks),
            sizes=tuple(sizes)
        )

    def pool(self, 
             x: torch.Tensor, 
             pool_type: Optional[str] = None,
             padding_mask: Optional[torch.Tensor] = None, # (B, S) with 0 for valid, 1 for padding
             size: Optional[torch.Tensor] = None # ToMe token size vector
            ) -> torch.Tensor:
        if self.attn_pool is not None:
            if padding_mask is not None:
                # Expand padding mask to match attention mask shape
                attn_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # Shape: (b, 1, 1, s)
                # Convert padding positions to -inf
                attn_mask = attn_mask * torch.finfo(x.dtype).min
            else:
                attn_mask = None
            x = self.attn_pool(x, attention_mask=attn_mask, size=size)
            return x
        pool_type = self.global_pool if pool_type is None else pool_type
        x = global_pool_nlc_w_masking(x, pool_type=pool_type, num_prefix_tokens=self.num_prefix_tokens, padding_mask=padding_mask)
        return x

    def forward_head(self, 
                     x: torch.Tensor, 
                     pre_logits: bool = False,
                     padding_mask: Optional[torch.Tensor] = None,
                     size: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.pool(x, padding_mask=padding_mask, size=size)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, padding_mask, final_size = self.forward_features(x)
        x = self.forward_head(x, padding_mask=padding_mask, size=final_size)
        return x

from timm.models.vision_transformer import VisionTransformer, checkpoint_filter_fn, build_model_with_cfg
def _create_tome_vision_transformer(variant: str, pretrained: bool = False, **kwargs) -> ToMEVisionTransformer:
    out_indices = kwargs.pop('out_indices', 3)
    if 'flexi' in variant:
        # FIXME Google FlexiViT pretrained models have a strong preference for bilinear patch / embed
        # interpolation, other pretrained models resize better w/ anti-aliased bicubic interpolation.
        _filter_fn = partial(checkpoint_filter_fn, interpolation='bilinear', antialias=False)
    else:
        _filter_fn = checkpoint_filter_fn

    # FIXME attn pool (currently only in siglip) params removed if pool disabled, is there a better soln?
    strict = kwargs.pop('pretrained_strict', True)
    if 'siglip' in variant and kwargs.get('global_pool', None) != 'map':
        strict = False
    print("Create ToME Vision Transformer with kwargs: ", kwargs)
    return build_model_with_cfg(
        ToMEVisionTransformer,
        variant,
        pretrained,
        pretrained_filter_fn=_filter_fn,
        pretrained_strict=strict,
        feature_cfg=dict(out_indices=out_indices, feature_cls='getter'),
        **kwargs,
    )


@register_model
def vit_base_patch16_siglip_384_tome_no_merge(pretrained: bool = False, **kwargs) -> VisionTransformer:
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, class_token=False, global_pool='map',
        merge_mode="batch_level", r_total=0, r_schedule="constant"
    ) # no merge; 
    # Override specific keys from kwargs if provided
    for key in ("r_total", "r_schedule", "merge_mode"):
        if key in kwargs:
            model_args[key] = kwargs.pop(key)

    model = _create_tome_vision_transformer(
        'vit_base_patch16_siglip_384', 
        pretrained=pretrained, 
        block_fn=ToMEBlock, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch16_siglip_224_tome(pretrained: bool = False, **kwargs) -> VisionTransformer:
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, class_token=False, global_pool='map',
        merge_mode="batch_level", r_total=12*8, r_schedule="constant"
    ) # original # tokens = 14*14 = 196 ; remove 12*8 = 96 tokens

    # Override specific keys from kwargs if provided
    for key in ("r_total", "r_schedule", "merge_mode"):
        if key in kwargs:
            model_args[key] = kwargs.pop(key)

    model = _create_tome_vision_transformer(
        'vit_base_patch16_siglip_224', 
        pretrained=pretrained, 
        block_fn=ToMEBlock, **dict(model_args, **kwargs))
    return model


@register_model # version used in llava1.5
def vit_large_patch14_clip_336_tome(pretrained: bool = False, **kwargs) -> VisionTransformer:
    # For CLIP ViT-L/14 at 336x336 resolution:
    # - patch_size: 14 (since 336/14 = 24 patches per side => 576 tokens)
    # - embed_dim: 1024, depth: 24, num_heads: 16 (as in CLIP’s large model)
    # - We remove 8 tokens per block: 24 * 8 = 192 tokens in total. TBD
    model_args = dict(
        img_size=336,
        patch_size=14, embed_dim=1024, depth=24, num_heads=16,
        # class_token=False,
        # global_pool='map',
        class_token=True,
        global_pool="token",
        pre_norm=True,
        merge_mode="batch_level",
        r_total=24 * 8, # total tokens to remove (192)
        r_schedule="constant"
    )

    # Allow overriding specific keys from kwargs if provided
    for key in ("r_total", "r_schedule", "merge_mode"):
        if key in kwargs:
            model_args[key] = kwargs.pop(key)

    model = _create_tome_vision_transformer(
        'vit_large_patch14_clip_336',
        pretrained=pretrained,
        block_fn=ToMEBlock, **dict(model_args, **kwargs))
    return model



### no merge classes for baseline llava ###
@register_model
def vit_base_patch16_siglip_384_tome_480out(pretrained: bool = False, **kwargs) -> VisionTransformer:
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, class_token=False, global_pool='map',
        merge_mode="batch_level", r_total=12*8, r_schedule="constant"
    ) # original # tokens = 576 ; remove 12*32 = 384 tokens

    # Override specific keys from kwargs if provided
    for key in ("r_total", "r_schedule", "merge_mode"):
        if key in kwargs:
            model_args[key] = kwargs.pop(key)

    model = _create_tome_vision_transformer(
        'vit_base_patch16_siglip_384', 
        pretrained=pretrained, 
        block_fn=ToMEBlock, **dict(model_args, **kwargs))
    return model




@register_model
def vit_base_patch16_siglip_384_tome_384out(pretrained: bool = False, **kwargs) -> VisionTransformer:
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, class_token=False, global_pool='map',
        merge_mode="batch_level", r_total=12*16, r_schedule="constant"
    ) # original # tokens = 576 ; remove 12*32 = 384 tokens

    # Override specific keys from kwargs if provided
    for key in ("r_total", "r_schedule", "merge_mode"):
        if key in kwargs:
            model_args[key] = kwargs.pop(key)

    model = _create_tome_vision_transformer(
        'vit_base_patch16_siglip_384', 
        pretrained=pretrained, 
        block_fn=ToMEBlock, **dict(model_args, **kwargs))
    return model





@register_model
def vit_base_patch16_siglip_384_tome_192out(pretrained: bool = False, **kwargs) -> VisionTransformer:
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, class_token=False, global_pool='map',
        merge_mode="batch_level", r_total=12*32, r_schedule="constant"
    )
    # Override specific keys from kwargs if provided
    for key in ("r_total", "r_schedule", "merge_mode"):
        if key in kwargs:
            model_args[key] = kwargs.pop(key)

    model = _create_tome_vision_transformer(
        'vit_base_patch16_siglip_384', 
        pretrained=pretrained, 
        block_fn=ToMEBlock, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch16_siglip_384_tome_72out(pretrained: bool = False, **kwargs) -> VisionTransformer:
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, class_token=False, global_pool='map',
        merge_mode="batch_level", r_total=12*42, r_schedule="constant"
    )
    # Override specific keys from kwargs if provided
    for key in ("r_total", "r_schedule", "merge_mode"):
        if key in kwargs:
            model_args[key] = kwargs.pop(key)

    model = _create_tome_vision_transformer(
        'vit_base_patch16_siglip_384', 
        pretrained=pretrained, 
        block_fn=ToMEBlock, **dict(model_args, **kwargs))
    return model



@register_model
def vit_base_patch16_siglip_384_tome_72out_linear(pretrained: bool = False, **kwargs) -> VisionTransformer:
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, class_token=False, global_pool='map',
        merge_mode="batch_level", r_total=12*42, r_schedule="linear"
    )
    # Override specific keys from kwargs if provided
    for key in ("r_total", "r_schedule", "merge_mode"):
        if key in kwargs:
            model_args[key] = kwargs.pop(key)

    model = _create_tome_vision_transformer(
        'vit_base_patch16_siglip_384', 
        pretrained=pretrained, 
        block_fn=ToMEBlock, **dict(model_args, **kwargs))
    return model


# for llava-one-vision
@register_model
def vit_so400m_patch14_siglip_384_tome_192out(pretrained: bool = False, **kwargs) -> VisionTransformer:
    model_args = dict(
        patch_size=14, embed_dim=1152, depth=27, num_heads=16, mlp_ratio=3.7362,
        class_token=False, global_pool='map',
        merge_mode="batch_level", r_total=537, r_schedule="constant"
    )
    # original 729 -> 192
    for key in ("r_total", "r_schedule", "merge_mode"):
        if key in kwargs:
            model_args[key] = kwargs.pop(key)

    model = _create_tome_vision_transformer(
        'vit_so400m_patch14_siglip_384', 
        pretrained=pretrained, 
        block_fn=ToMEBlock, **dict(model_args, **kwargs))
    return model