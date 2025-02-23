from collections import OrderedDict
import math
from typing import Callable, List, Optional, Sequence, Tuple, Union
from functools import partial

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from open_clip.tome import batch_level_bipartite_soft_matching, batch_level_merge_wavg, bipartite_soft_matching, merge_source, merge_wavg, CLIPVisionEncoderToMEOutput

from .utils import to_2tuple
from .pos_embed import get_2d_sincos_pos_embed


class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            scaled_cosine: bool = False,
            scale_heads: bool = False,
            logit_scale_max: float = math.log(1. / 0.01),
            batch_first: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logit_scale_max = logit_scale_max
        self.batch_first = batch_first
        self.use_fsdpa = hasattr(nn.functional, 'scaled_dot_product_attention')

        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
        self.in_proj_weight = nn.Parameter(torch.randn((dim * 3, dim)) * self.scale)
        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
        else:
            self.in_proj_bias = None

        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)
        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        if self.batch_first:
            x = x.transpose(0, 1)

        L, N, C = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.reshape(L, N * self.num_heads, -1).transpose(0, 1)
        k = k.reshape(L, N * self.num_heads, -1).transpose(0, 1)
        v = v.reshape(L, N * self.num_heads, -1).transpose(0, 1)

        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask

        if self.logit_scale is not None:
            attn = torch.bmm(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2))
            logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn.view(N, self.num_heads, L, L) * logit_scale
            attn = attn.view(-1, L, L)
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = torch.bmm(attn, v)
        else:
            if self.use_fsdpa:
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )
            else:
                q = q * self.scale
                attn = torch.bmm(q, k.transpose(-1, -2))
                if attn_mask is not None:
                    attn += attn_mask
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = torch.bmm(attn, v)

        if self.head_scale is not None:
            x = x.view(N, self.num_heads, L, C) * self.head_scale
            x = x.view(-1, L, C)

        x = x.transpose(0, 1).reshape(L, N, C)

        if self.batch_first:
            x = x.transpose(0, 1)

        x = self.out_proj(x)
        x = self.out_drop(x)
        return x


class AttentionalPooler(nn.Module):
    def __init__(
            self,
            d_model: int,
            context_dim: int,
            n_head: int = 8,
            n_queries: int = 256,
            norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_head, kdim=context_dim, vdim=context_dim, batch_first=True)
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

    def forward(self, x: torch.Tensor):
        N = x.shape[0]
        x = self.ln_k(x)
        q = self.ln_q(self.query)
        out = self.attn(q.unsqueeze(0).expand(N, -1, -1), x, x, need_weights=False)[0]
        return out


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            is_cross_attention: bool = False,
            batch_first: bool = True,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=batch_first)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(
            q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask
        )[0]

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class ToMEMultiheadAttention(nn.MultiheadAttention):
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        size:Optional[Tensor]=None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None):
        assert self.batch_first
        B = query.shape[0]
        N = query.shape[1]
        full_bias = None
        if size is not None:
            size_bias_log = size.log()[:, :, 0] # (b, src_len, 1) -> (b, src_len)
            size_bias_log = size_bias_log.unsqueeze(1).unsqueeze(1).expand(B, self.num_heads, N, N) # (b, src_len) -> (b, num_heads, 1, src_len)
            full_bias = size_bias_log
        
        # apply attention mask before softmax (-inf for masked tokens)
        if attn_mask is not None:
            if attn_mask.size() != (B, 1, N, N):
                raise ValueError(
                    f"Attention mask should be of size {(B, 1, N, N)}, but is {attn_mask.size()}"
                )
            # Expand the attn_mask to [B, num_heads, N, N]
            expanded_mask = attn_mask.expand(B, self.num_heads, N, N)
            if full_bias is None:
                full_bias = expanded_mask
            else:
                full_bias = full_bias + expanded_mask

        # If a bias mask was created, reshape it to merge the batch and head dimensions.
        if full_bias is not None:
            full_bias = full_bias.reshape(B * self.num_heads, N, N)        

        return super().forward(query=query, key=key, value=value, attn_mask=full_bias, need_weights=need_weights)



class ToMEResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            is_cross_attention: bool = False,
            batch_first: bool = True,
            # TOME args
            trace_source: bool = False,
            prop_attn: bool = True,
            cls_token: bool = True,
            r: int = 0,
            merge_mode: str = "instance_level",
            max_r_per_instance_ratio: float = None,
            update_threshold: bool = False,
            specified_threshold: float = None

    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = ToMEMultiheadAttention(d_model, n_head, batch_first=batch_first)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
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
        self.momentum = 0.1
        self.specified_threshold = specified_threshold

    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        attn_out = self.attn(
            q_x, k_x, v_x, need_weights=False, size=attn_size, attn_mask=attn_mask
        )[0]
        metric = k_x
        return attn_out, metric

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
        else:
            raise ValueError(f"Invalid merge mode: {self._tome_info['merge_mode']}; must be 'instance_level' or 'batch_level'")
            
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


    def forward(
            self,
            q_x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
            padding_mask: Optional[torch.Tensor] = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        dtype = q_x.dtype
        assert attn_mask is None
        if padding_mask is not None:
            attn_mask = self._get_attn_mask_from_padding_mask(padding_mask, dtype)
        else:
            attn_mask = None
        attn_out, metric = self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        x = q_x + self.ls_1(attn_out)
        
        r = self._tome_info["r"]
        if r > 0:
            x, padding_mask = self.merge_tokens(metric, r, x, padding_mask=padding_mask)
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x, padding_mask




class CustomResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            scale_cosine_attn: bool = False,
            scale_heads: bool = False,
            scale_attn: bool = False,
            scale_fc: bool = False,
            batch_first: bool = True,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = Attention(
            d_model,
            n_head,
            scaled_cosine=scale_cosine_attn,
            scale_heads=scale_heads,
            batch_first=batch_first,
        )
        self.ln_attn = norm_layer(d_model) if scale_attn else nn.Identity()
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ('ln', norm_layer(mlp_width) if scale_fc else nn.Identity()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def get_reference_weight(self):
        return self.mlp.c_fc.weight

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.ls_1(self.ln_attn(self.attn(self.ln_1(x), attn_mask=attn_mask)))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)


class Transformer(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            batch_first: bool = True,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.batch_first = batch_first
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                batch_first=batch_first,
            )
            for _ in range(layers)
        ])

    def get_cast_dtype(self) -> torch.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, 'int8_original_dtype'):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()    # NLD -> LND
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        if not self.batch_first:
            x = x.transpose(0, 1)    # LND -> NLD
        return x


class CustomTransformer(nn.Module):
    """ A custom transformer that can use different block types. """
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            batch_first: bool = True,
            block_types: Union[str, List[str]] = 'CustomResidualAttentionBlock',
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.batch_first = batch_first  # run trasnformer stack in batch first (N, L, D)
        self.grad_checkpointing = False

        if isinstance(block_types, str):
            block_types = [block_types] * layers
        assert len(block_types) == layers

        def _create_block(bt: str):
            if bt == 'CustomResidualAttentionBlock':
                return CustomResidualAttentionBlock(
                    width,
                    heads,
                    mlp_ratio=mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    batch_first=batch_first,
                )
            else:
                assert False

        self.resblocks = nn.ModuleList([
            _create_block(bt)
            for bt in block_types
        ])

    def get_cast_dtype(self) -> torch.dtype:
        weight = self.resblocks[0].get_reference_weight()
        if hasattr(weight, 'int8_original_dtype'):
            return weight.int8_original_dtype
        return weight.dtype

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        if not self.batch_first:
            x = x.transpose(0, 1)  # NLD -> LND

        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)

        if not self.batch_first:
            x = x.transpose(0, 1)  # NLD -> LND
        return x




class ToMEOpenAITransformer(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            batch_first: bool = True,
            # tome args
            merge_mode: str = "batch_level", # merge mode: instance_level or batch_level
            r_total: int = 0, # total number of tokens to remove
            r_schedule: str = "constant", # r schedule: constant, linear, reverse_linear
            max_r_per_instance_ratio: float = None, # 1.0 => rever to fixed r for each instance; > 1.0 => dynamic r
            update_threshold: bool = False, # whether to post-hoc update threshold after training
            specified_thresholds: List[float] = None, # specified threshold for each layer
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.batch_first = batch_first
        self.grad_checkpointing = False
        ### ToMe configs ###
        self.rs = self._get_rs(layers, r_total, r_schedule)
        print("set total avg remove token nums each layer as: ", self.rs)
        print("merge mode: ", merge_mode)
        assert merge_mode in ["instance_level", "batch_level"]
        self._tome_info = {
            "size": None,
            "source": None,
            "trace_source": False,
            "prop_attn": True,
            "class_token": True,
            "distill_token": False,
            "merge_mode": merge_mode,
            "max_r_per_instance_ratio": max_r_per_instance_ratio,
            "update_threshold": update_threshold,
            "specified_thresholds": specified_thresholds,
        }
        self.resblocks = nn.ModuleList([
            ToMEResidualAttentionBlock(
                width,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                batch_first=batch_first,
                ## ToMe args
                trace_source = False,
                prop_attn = True,
                cls_token = True,
                r = self.rs[i],
                merge_mode = merge_mode,
                max_r_per_instance_ratio = max_r_per_instance_ratio,
                update_threshold = update_threshold,
                specified_threshold = specified_thresholds[i] if specified_thresholds is not None else None
            )
            for i in range(layers)
        ])

    def get_cast_dtype(self) -> torch.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, 'int8_original_dtype'):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype

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


    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        # for enable getting intermediate features for llava-style model
        self._tome_info["size"] = None
        self._tome_info["source"] = None
        assert self.batch_first, "only support batch_first"
        # if not self.batch_first:
        #     x = x.transpose(0, 1).contiguous()    # NLD -> LND

        hidden_states = []
        padding_masks = []
        sizes = []
        self.output_stats = {}
        padding_mask = None
        for idx, r in enumerate(self.resblocks):
            r._tome_info["size"] = self._tome_info["size"]
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x, padding_mask = checkpoint(r, x, None, None, attn_mask, padding_mask)
            else:
                x, padding_mask = r(x, attn_mask=attn_mask, padding_mask=padding_mask)
            self._tome_info["size"] = r._tome_info["size"]
            hidden_states.append(x)
            padding_masks.append(padding_mask)
            sizes.append(self._tome_info["size"])
            if padding_mask is not None:
                ntoks = (padding_mask<0.5).float().sum(-1)
                ntoks = ntoks.detach().tolist()
            else:
                ntoks = [x.shape[1]]*x.shape[0]
            self.output_stats[f"block_{idx}_ntoks"] = ntoks
        final_size = self._tome_info["size"]

        # if not self.batch_first:
        #     x = x.transpose(0, 1)    # LND -> NLD
        return x, padding_mask, final_size, {"hidden_states": hidden_states, "padding_masks": padding_masks, "sizes": sizes}


class VisionTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            ls_init_value: float = None,
            attentional_pool: bool = False,
            attn_pooler_queries: int = 256,
            attn_pooler_heads: int = 8,
            output_dim: int = 512,
            patch_dropout: float = 0.,
            no_ln_pre: bool = False,
            pos_embed_type: str = 'learnable',
            pool_type: str = 'tok',
            final_ln_after_pool: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False,
    ):
        super().__init__()
        assert pool_type in ('tok', 'avg', 'none')
        self.output_tokens = output_tokens
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.final_ln_after_pool = final_ln_after_pool  # currently ignored w/ attn pool enabled
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        # class embeddings and positional embeddings
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        if pos_embed_type == 'learnable':
            self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        elif pos_embed_type == 'sin_cos_2d':
            # fixed sin-cos embedding
            assert self.grid_size[0] == self.grid_size[1],\
                'currently sin cos 2d pos embedding only supports square input'
            self.positional_embedding = nn.Parameter(
                torch.zeros(self.grid_size[0] * self.grid_size[1] + 1, width), requires_grad=False)
            pos_embed_type = get_2d_sincos_pos_embed(width, self.grid_size[0], cls_token=True)
            self.positional_embedding.data.copy_(torch.from_numpy(pos_embed_type).float())
        else:
            raise ValueError

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()

        self.ln_pre = nn.Identity() if no_ln_pre else norm_layer(width)
        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        if attentional_pool:
            if isinstance(attentional_pool, str):
                self.attn_pool_type = attentional_pool
                self.pool_type = 'none'
                if attentional_pool in ('parallel', 'cascade'):
                    self.attn_pool = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=attn_pooler_queries,
                    )
                    self.attn_pool_contrastive = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=1,
                    )
                else:
                    assert False
            else:
                self.attn_pool_type = ''
                self.pool_type = pool_type
                self.attn_pool = AttentionalPooler(
                    output_dim,
                    width,
                    n_head=attn_pooler_heads,
                    n_queries=attn_pooler_queries,
                )
                self.attn_pool_contrastive = None
            pool_dim = output_dim
        else:
            self.attn_pool = None
            pool_dim = width
            self.pool_type = pool_type

        self.ln_post = norm_layer(pool_dim)
        self.proj = nn.Parameter(scale * torch.randn(pool_dim, output_dim))

        self.init_parameters()

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        for param in self.parameters():
            param.requires_grad = False

        if unlocked_groups != 0:
            groups = [
                [
                    self.conv1,
                    self.class_embedding,
                    self.positional_embedding,
                    self.ln_pre,
                ],
                *self.transformer.resblocks[:-1],
                [
                    self.transformer.resblocks[-1],
                    self.ln_post,
                ],
                self.proj,
            ]

            def _unlock(x):
                if isinstance(x, Sequence):
                    for g in x:
                        _unlock(g)
                else:
                    if isinstance(x, torch.nn.Parameter):
                        x.requires_grad = True
                    else:
                        for p in x.parameters():
                            p.requires_grad = True

            _unlock(groups[-unlocked_groups:])

    def init_parameters(self):
        # FIXME OpenAI CLIP did not define an init for the VisualTransformer
        # TODO experiment if default PyTorch init, below, or alternate init is best.

        # nn.init.normal_(self.class_embedding, std=self.scale)
        # nn.init.normal_(self.positional_embedding, std=self.scale)
        #
        # proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        # attn_std = self.transformer.width ** -0.5
        # fc_std = (2 * self.transformer.width) ** -0.5
        # for block in self.transformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        #
        # if self.text_projection is not None:
        #     nn.init.normal_(self.text_projection, std=self.scale)
        pass

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    @torch.jit.ignore
    def no_weight_decay(self):
        # for timm optimizers, 1d params like logit_scale, logit_bias, ln/bn scale, biases are excluded by default
        no_wd = {'positional_embedding', 'class_embedding'}
        return no_wd

    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pool_type == 'avg':
            pooled, tokens = x[:, 1:].mean(dim=1), x[:, 1:]
        elif self.pool_type == 'tok':
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled = tokens = x

        return pooled, tokens

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.patch_dropout(x)
        x = self.ln_pre(x)
        x = self.transformer(x)

        if self.attn_pool is not None:
            if self.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.attn_pool(x)
                if self.attn_pool_type == 'parallel':
                    pooled = self.attn_pool_contrastive(x)
                else:
                    assert self.attn_pool_type == 'cascade'
                    pooled = self.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.attn_pool(x)
                x = self.ln_post(x)
                pooled, tokens = self._global_pool(x)
        elif self.final_ln_after_pool:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)
        else:
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)

        if self.proj is not None:
            pooled = pooled @ self.proj

        if self.output_tokens:
            return pooled, tokens
        
        return pooled


class AttentionalPoolerWMasking(nn.Module):
    def __init__(
            self,
            d_model: int,
            context_dim: int,
            n_head: int = 8,
            n_queries: int = 256,
            norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_head, kdim=context_dim, vdim=context_dim, batch_first=True)
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

    def forward(self, 
                x: torch.Tensor,  
                attention_mask: Optional[torch.Tensor] = None,
                size: Optional[int] = None
                ):
        B = x.shape[0]
        assert len(x.shape) == 3, f"Input shape should be [B, L, D], got {x.shape}"
        x = self.ln_k(x)
        q = self.ln_q(self.query)

        full_bias = None
        # If size is provided, compute a bias from it.
        if size is not None:
            # Assume size has shape [B, L, 1]
            # Log transform and squeeze the last dim to get [B, L]
            size_bias_log = size.log()[:, :, 0]
            # Expand to [B, n_queries, L]
            size_bias_log = size_bias_log.unsqueeze(1).expand(B, q.shape[0], x.shape[1])
            full_bias = size_bias_log
        
        # Incorporate the attention_mask if provided.
        if attention_mask is not None:
            # Expect attention_mask to be [B, 1, L]
            assert attention_mask.size() == (B, 1, x.shape[1]), (
                f"Attention mask shape {attention_mask.size()} not compatible with expected shape {(B, 1, x.shape[1])}"
            )
            # Expand to [B, n_queries, L]
            attention_mask = attention_mask.expand(B, q.shape[0], x.shape[1])
            if full_bias is None:
                full_bias = attention_mask
            else:
                full_bias = full_bias + attention_mask
        
        # If we have a mask (or bias), reshape it for multi-head attention.
        if full_bias is not None:
            # full_bias is currently [B, n_queries, L]
            full_bias = full_bias.unsqueeze(1)  # -> [B, 1, n_queries, L]
            full_bias = full_bias.expand(B, self.n_head, q.shape[0], x.shape[1])  # -> [B, n_head, n_queries, L]
            # Reshape to merge the batch and head dimensions: [B * n_head, n_queries, L]
            full_bias = full_bias.reshape(B * self.n_head, q.shape[0], x.shape[1])

        out = self.attn(q.unsqueeze(0).expand(B, -1, -1), x, x, attn_mask=full_bias, need_weights=False)[0]
        return out



class ToMEOpenAIVisionTransformer(VisionTransformer):

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            ls_init_value: float = None,
            attentional_pool: bool = False,
            attn_pooler_queries: int = 256,
            attn_pooler_heads: int = 8,
            output_dim: int = 512,
            patch_dropout: float = 0.,
            no_ln_pre: bool = False,
            pos_embed_type: str = 'learnable',
            pool_type: str = 'tok',
            final_ln_after_pool: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False,
            # tome args
            merge_mode: str = "batch_level", # merge mode: instance_level or batch_level
            r_total: int = 0, # total number of tokens to remove
            r_schedule: str = "constant", # r schedule: constant, linear, reverse_linear
            max_r_per_instance_ratio: float = None, # 1.0 => rever to fixed r for each instance; > 1.0 => dynamic r
            update_threshold: bool = False, # whether to post-hoc update threshold after training
            specified_thresholds: List[float] = None, # specified threshold for each layer
            **kwargs
    ):
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            attentional_pool=attentional_pool,
            attn_pooler_queries=attn_pooler_queries,
            attn_pooler_heads=attn_pooler_heads,
            output_dim=output_dim,
            patch_dropout=patch_dropout,
            no_ln_pre=no_ln_pre,
            pos_embed_type=pos_embed_type,
            pool_type=pool_type,
            final_ln_after_pool=final_ln_after_pool,
            act_layer=act_layer,
            norm_layer=norm_layer,
            output_tokens=output_tokens,
            **kwargs
        )
        del self.transformer

        ###
        self.transformer = ToMEOpenAITransformer(width, layers, heads, mlp_ratio, ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,merge_mode=merge_mode, r_total=r_total, r_schedule=r_schedule, max_r_per_instance_ratio=max_r_per_instance_ratio, update_threshold=update_threshold, specified_thresholds=specified_thresholds)

        ###
        if attentional_pool:
            if isinstance(attentional_pool, str):
                self.attn_pool_type = attentional_pool
                self.pool_type = 'none'
                if attentional_pool in ('parallel', 'cascade'):
                    self.attn_pool = AttentionalPoolerWMasking(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=attn_pooler_queries,
                    )
                    self.attn_pool_contrastive = AttentionalPoolerWMasking(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=1,
                    )
                else:
                    assert False
            else:
                self.attn_pool_type = ''
                self.pool_type = pool_type
                self.attn_pool = AttentionalPoolerWMasking(
                    output_dim,
                    width,
                    n_head=attn_pooler_heads,
                    n_queries=attn_pooler_queries,
                )
                self.attn_pool_contrastive = None
            pool_dim = output_dim
        else:
            self.attn_pool = None
            pool_dim = width
            self.pool_type = pool_type
        
        print("attn_pool:", self.attn_pool)
        print("pool_type:", self.pool_type)


    def _global_pool_w_masking(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None  # (B, S) with 0 for valid, 1 for padding
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies global pooling while considering padding masks.

        Args:
            x (torch.Tensor): Input tensor of shape (B, S, D).
            self.pool_type (str): Pooling type - 'avg' for average pooling, 'tok' for token-based pooling.
            padding_mask (Optional[torch.Tensor]): Padding mask of shape (B, S), where 0 is valid and 1 is padding.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Pooled tensor and token tensor.
        """
        if self.pool_type == 'avg':
            tokens = x[:, 1:]  # Exclude CLS token
            if padding_mask is not None:
                mask = padding_mask[:, 1:].unsqueeze(-1)  # (B, S-1, 1)
                valid_count = (mask == 0).sum(dim=1, keepdim=True).clamp(min=1)  # Avoid division by zero
                pooled = (tokens * (mask == 0)).sum(dim=1) / valid_count.squeeze(1)
            else:
                pooled = tokens.mean(dim=1)
        elif self.pool_type == 'tok':
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled = tokens = x

        return pooled, tokens

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.patch_dropout(x)
        x = self.ln_pre(x)
        x, padding_mask, final_size, additional_info = self.transformer(x)
        self.output_stats = self.transformer.output_stats
        
        # get 3D attn mask
        if padding_mask is not None:
            # Expand padding mask to match attention mask shape
            attn_mask = padding_mask.unsqueeze(1)  # Shape: (b, 1, s)
            # Convert padding positions to -inf
            attn_mask = attn_mask * torch.finfo(x.dtype).min
        else:
            attn_mask = None

        # assert False, "Need to fix this pooling method for tome"
        if self.attn_pool is not None:
            if self.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.attn_pool(x, attention_mask=attn_mask, size=final_size)
                if self.attn_pool_type == 'parallel':
                    pooled = self.attn_pool_contrastive(x, attention_mask=attn_mask, size=final_size)
                else:
                    assert self.attn_pool_type == 'cascade'
                    pooled = self.attn_pool_contrastive(tokens, attention_mask=attn_mask, size=final_size)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.attn_pool(x, attention_mask=attn_mask, size=final_size)
                x = self.ln_post(x)
                pooled, tokens = self._global_pool_w_masking(x, padding_mask)
        elif self.final_ln_after_pool:
            pooled, tokens = self._global_pool_w_masking(x, padding_mask)
            pooled = self.ln_post(pooled)
        else:
            x = self.ln_post(x)
            pooled, tokens = self._global_pool_w_masking(x, padding_mask)

        if self.proj is not None:
            pooled = pooled @ self.proj

        if self.output_tokens:
            return pooled, tokens
        
        return pooled

    def forward_features_all_layers(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.patch_dropout(x)
        x = self.ln_pre(x)
        x, padding_mask, final_size, additional_info = self.transformer(x)
        self.output_stats = self.transformer.output_stats
        
        # get 3D attn mask
        if padding_mask is not None:
            # Expand padding mask to match attention mask shape
            attn_mask = padding_mask.unsqueeze(1)  # Shape: (b, 1, s)
            # Convert padding positions to -inf
            attn_mask = attn_mask * torch.finfo(x.dtype).min
        else:
            attn_mask = None

        # assert False, "Need to fix this pooling method for tome"
        if self.attn_pool is not None:
            if self.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.attn_pool(x, attention_mask=attn_mask, size=final_size)
                if self.attn_pool_type == 'parallel':
                    pooled = self.attn_pool_contrastive(x, attention_mask=attn_mask, size=final_size)
                else:
                    assert self.attn_pool_type == 'cascade'
                    pooled = self.attn_pool_contrastive(tokens, attention_mask=attn_mask, size=final_size)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.attn_pool(x, attention_mask=attn_mask, size=final_size)
                x = self.ln_post(x)
                pooled, tokens = self._global_pool_w_masking(x, padding_mask)
        elif self.final_ln_after_pool:
            pooled, tokens = self._global_pool_w_masking(x, padding_mask)
            pooled = self.ln_post(pooled)
        else:
            x = self.ln_post(x)
            pooled, tokens = self._global_pool_w_masking(x, padding_mask)

        if self.proj is not None:
            pooled = pooled @ self.proj

        if self.output_tokens:
            return pooled, tokens
        
        return CLIPVisionEncoderToMEOutput(
            last_hidden_state=x,
            pooler_output=pooled,
            hidden_states=tuple(additional_info['hidden_states']),
            padding_masks=tuple(additional_info['padding_masks']),
            sizes=tuple(additional_info['sizes'])
        )
        

def text_global_pool(x, text: Optional[torch.Tensor] = None, pool_type: str = 'argmax'):
    if pool_type == 'first':
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == 'last':
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == 'argmax':
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
    else:
        pooled = tokens = x

    return pooled, tokens


class TextTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            context_length: int = 77,
            vocab_size: int = 49408,
            width: int = 512,
            heads: int = 8,
            layers: int = 12,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            output_dim: Optional[int] = 512,
            embed_cls: bool = False,
            no_causal_mask: bool = False,
            pad_id: int = 0,
            pool_type: str = 'argmax',
            proj_type: str = 'linear',
            proj_bias: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False,
    ):
        super().__init__()
        assert pool_type in ('first', 'last', 'argmax', 'none')
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pad_id = pad_id
        self.pool_type = pool_type

        self.token_embedding = nn.Embedding(vocab_size, width)
        if embed_cls:
            self.cls_emb = nn.Parameter(torch.empty(width))
            self.num_pos += 1
        else:
            self.cls_emb = None
        self.positional_embedding = nn.Parameter(torch.empty(self.num_pos, width))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.ln_final = norm_layer(width)

        if no_causal_mask:
            self.attn_mask = None
        else:
            self.register_buffer('attn_mask', self.build_causal_mask(), persistent=False)

        if proj_type == 'none' or not output_dim:
            self.text_projection = None
        else:
            if proj_bias:
                self.text_projection = nn.Linear(width, output_dim)
            else:
                self.text_projection = nn.Parameter(torch.empty(width, output_dim))

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if self.cls_emb is not None:
            nn.init.normal_(self.cls_emb, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                nn.init.normal_(self.text_projection.weight, std=self.transformer.width ** -0.5)
                if self.text_projection.bias is not None:
                    nn.init.zeros_(self.text_projection.bias)
            else:
                nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    @torch.jit.ignore
    def no_weight_decay(self):
        # for timm optimizers, 1d params like logit_scale, logit_bias, ln/bn scale, biases are excluded by default
        no_wd = {'positional_embedding'}
        if self.cls_emb is not None:
            no_wd.add('cls_emb')
        return no_wd

    def build_causal_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def build_cls_mask(self, text, cast_dtype: torch.dtype):
        cls_mask = (text != self.pad_id).unsqueeze(1)
        cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=True)
        additive_mask = torch.empty(cls_mask.shape, dtype=cast_dtype, device=cls_mask.device)
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float("-inf"))
        additive_mask = torch.repeat_interleave(additive_mask, self.heads, 0)
        return additive_mask

    def forward(self, text):
        cast_dtype = self.transformer.get_cast_dtype()
        seq_len = text.shape[1]

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        attn_mask = self.attn_mask
        if self.cls_emb is not None:
            seq_len += 1
            x = torch.cat([x, _expand_token(self.cls_emb, x.shape[0])], dim=1)
            cls_mask = self.build_cls_mask(text, cast_dtype)
            if attn_mask is not None:
                attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]

        x = x + self.positional_embedding[:seq_len].to(cast_dtype)
        x = self.transformer(x, attn_mask=attn_mask)

        # x.shape = [batch_size, n_ctx, transformer.width]
        if self.cls_emb is not None:
            # presence of appended cls embed (CoCa) overrides pool_type, always take last token
            pooled, tokens = text_global_pool(x, pool_type='last')
            pooled = self.ln_final(pooled)  # final LN applied after pooling in this case
        else:
            x = self.ln_final(x)
            pooled, tokens = text_global_pool(x, text, pool_type=self.pool_type)

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                pooled = self.text_projection(pooled)
            else:
                pooled = pooled @ self.text_projection

        if self.output_tokens:
            return pooled, tokens

        return pooled


class MultimodalTransformer(Transformer):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            context_length: int = 77,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_dim: int = 512,
            batch_first: bool = True,
    ):
        super().__init__(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            batch_first=batch_first,
        )
        self.context_length = context_length
        self.cross_attn = nn.ModuleList([
            ResidualAttentionBlock(
                width,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                is_cross_attention=True,
                batch_first=batch_first,
            )
            for _ in range(layers)
        ])

        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

        self.ln_final = norm_layer(width)
        self.text_projection = nn.Parameter(torch.empty(width, output_dim))

    def init_parameters(self):
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        for block in self.transformer.cross_attn:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, image_embs, text_embs):
        seq_len = text_embs.shape[1]
        if not self.batch_first:
            image_embs = image_embs.permute(1, 0, 2)  # NLD -> LND
            text_embs = text_embs.permute(1, 0, 2)  # NLD -> LND

        for resblock, cross_attn in zip(self.resblocks, self.cross_attn):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                text_embs = checkpoint(resblock, text_embs, None, None, self.attn_mask[:seq_len, :seq_len])
                text_embs = checkpoint(cross_attn, text_embs, image_embs, image_embs, None)
            else:
                text_embs = resblock(text_embs, attn_mask=self.attn_mask[:seq_len, :seq_len])
                text_embs = cross_attn(text_embs, k_x=image_embs, v_x=image_embs)

        if not self.batch_first:
            text_embs = text_embs.permute(1, 0, 2)  # LND -> NLD

        out = self.ln_final(text_embs)
        if self.text_projection is not None:
            out = out @ self.text_projection

        return out

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable


