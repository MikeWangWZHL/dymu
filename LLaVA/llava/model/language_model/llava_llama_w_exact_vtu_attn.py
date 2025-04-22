## IMPORTANT NOTE: This version provides the exact VTU attention implementation `vtu_atten_kernal()` as described in the paper.
# The vtu attention results in lower FLOPs, but in practice, the wall clock time is much higher than sdpa.
# By default, we use the direct expansion + sdpa implementation in `llava_llama.py`.
# We encourage users to explore further optimizations in terms of wall clock time for VTU attention. 


#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import TYPE_CHECKING, List, Optional, Tuple, Union, Callable, Dict, Any

import ipdb
import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaAttention,
    ALL_ATTENTION_FUNCTIONS, 
    logger,
    eager_attention_forward
)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim) # (1,1,seq_len, head_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin) # q: (1, num_head, seq_len, head_dim)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

from torch_scatter import scatter_mean

def remerge_mapping_attn_out(Y: torch.Tensor, mapping_indices: torch.Tensor, N_un: int = None) -> torch.Tensor:
    return scatter_mean(Y, mapping_indices, dim=2, dim_size=N_un)

def remerge_mapping_hidden_states(X: torch.Tensor, mapping_indices: torch.Tensor, N_un: int) -> torch.Tensor:
    return scatter_mean(X, mapping_indices, dim=1, dim_size=N_un)


def vtu_atten_kernal(Q_un, K_un, cos, sin, mapping_indices, scaling):
    D = cos.shape[-1]
    C = cos[..., : D // 2]  # (1, N, D//2)
    S = sin[..., : D // 2]  # (1, N, D//2)
    
    # First, split the head-dimension into 2 halves.
    Q_un_1, Q_un_2 = torch.chunk(Q_un, 2, dim=-1)
    Q_un_reshape = torch.stack((Q_un_1, Q_un_2), dim=-2)  # (B, num_heads, N_un, 2, D//2)

    K_un_1, K_un_2 = torch.chunk(K_un, 2, dim=-1)
    K_un_reshape = torch.stack((K_un_1, K_un_2), dim=-2)  # (B, num_heads, N_un, 2, D//2)

    # Compute rotated version of K: (-K_un_2, K_un_1)
    K_un_rot_reshape = torch.stack((-K_un_2, K_un_1), dim=-2)  # (B, num_heads, N_un, 2, D//2)

    ### version 2
    Q_un_K_un_T = torch.einsum('b h i r d, b h r j d -> b h i j d', Q_un_reshape, K_un_reshape.transpose(2,3)) # (B, num_heads, N_un, N_un, D//2)
    Q_un_K_un_T_rot = torch.einsum('b h i r d, b h r j d -> b h i j d', Q_un_reshape, K_un_rot_reshape.transpose(2,3)) # (B, num_heads, N_un, N_un, D//2)
    # -------------------------------------------
    # 1.  gather the rows/cols you actually need
    # -------------------------------------------
    QK_dot   = Q_un_K_un_T     [:, :, mapping_indices][:, :, :, mapping_indices]   # (B, H, N, N, D/2)
    QK_cross = Q_un_K_un_T_rot[:, :, mapping_indices][:, :, :, mapping_indices]   # (B, H, N, N, D/2)

    # -------------------------------------------------
    # broadcast helpers: shapes → (1,1,N,1,D/2) and (1,1,1,N,D/2)
    Ci, Cj = C[0][None, None, :, None, :], C[0][None, None, None, :, :]
    Si, Sj = S[0][None, None, :, None, :], S[0][None, None, None, :, :]


    # -------------------------------------------------
    # build per‑(i,j,d) weights
    # -------------------------------------------------
    w_dot   = Ci * Cj + Si * Sj            # (1,1,N,N,D/2)
    w_cross = Ci * Sj - Si * Cj            # (1,1,N,N,D/2)  ←  sign fixed

    # -------------------------------------------------
    # fused reduction
    # -------------------------------------------------
    attn_scores = (
        (QK_dot   * w_dot  ).sum(dim=-1)   # (B,H,N,N)
        + (QK_cross * w_cross).sum(dim=-1)   # (B,H,N,N)
    ) * scaling                                          # (B, H, N, N)
    return attn_scores



class CustomLlamaAttention(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        mapping_indices: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        # customized attention
        assert self.config._attn_implementation == "eager", f"Only eager is supported for this exact VTU attention, but got {self.config._attn_implementation}; Please set `attn_implementation` to \"eager\" during model loading with .from_pretrained()."
        # hidden_states: (B, N_un, total_dim)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        Q_un = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2) # (B, num_heads, N_un, head_dim)
        K_un = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        V_un = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings # (1, N, head_dim)

        new_mapping_index = None # None for the first forward pass
        if mapping_indices is not None: # for the first forward pass during generation
            attn_scores = vtu_atten_kernal(Q_un, K_un, cos, sin, mapping_indices, self.scaling) # (B, num_heads, N, N)
            
            K_m = K_un.index_select(dim=2, index=mapping_indices) # (B, num_heads, N_un, D) -> (B, num_heads, N, D)
            V_m = V_un.index_select(dim=2, index=mapping_indices) # (B, num_heads, N_un, D) -> (B, num_heads, N, D)
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "mapping_indices": mapping_indices, "cache_position": cache_position}
                _,_,_,_,_ = past_key_value.update(K_m, V_m, self.layer_idx, cache_kwargs) # concat key, value and cos, sin for kv; updated mapping_indices
        else:
            # get previous mapping_indices
            if past_key_value is not None and len(past_key_value.mapping_indices) > self.layer_idx: # the second statement handles cases where we don't have mapping_indices from the beginning
                prev_mapping_indices = past_key_value.mapping_indices[self.layer_idx]
                if prev_mapping_indices is not None:
                    # find max
                    prev_un_len = prev_mapping_indices.max().item() + 1
                    new_mapping_index = torch.tensor([prev_un_len], device=prev_mapping_indices.device, dtype=prev_mapping_indices.dtype)
                    mapping_indices = new_mapping_index

            ## Update keys/values with cache if using one. saving expanded key values
            Q_m = Q_un
            K_m = K_un
            V_m = V_un
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "mapping_indices": mapping_indices, "cache_position": cache_position}
                K_m, V_m, cos_k, sin_k, mapping_indices = past_key_value.update(K_m, V_m, self.layer_idx, cache_kwargs) # concat key, value and cos, sin for kv; updated mapping_indices
            
            # apply rotary embedding
            cos_q = cos.unsqueeze(1) # (1,1,seq_len, head_dim)
            sin_q = sin.unsqueeze(1)
            cos_k = cos_k.unsqueeze(1) # (1,1,seq_len, head_dim)
            sin_k = sin_k.unsqueeze(1)
            q_embed = (Q_m * cos_q) + (rotate_half(Q_m) * sin_q) # q: (1, num_head, seq_len, head_dim)
            k_embed = (K_m * cos_k) + (rotate_half(K_m) * sin_k)
            attn_scores = torch.matmul(q_embed, k_embed.transpose(2, 3)) * self.scaling

        ## Handle attention mask
        if attention_mask is not None: # (1,1,N,N)
            attn_scores = attn_scores + attention_mask

        ## Compute attention weights and apply dropout.
        attn_weights = nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32).to(Q_un.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=0.0 if not self.training else self.attention_dropout)

        ## remerge steps: ##
        # Compute attention output in the compressed space.
        # f(e) = smax( A/√D ) V, with shape (B, num_heads, N, head_dim)
        attn_output = torch.matmul(attn_weights, V_m)

        # --- Map back to the original sequence length ---
        # f'(e) = (M^T M)^{-1} M^T f(e)
        # Using our sparse implementation, we "remerge" the compressed tensor back to length N_un.
        if new_mapping_index is None and mapping_indices is not None: # for the first forward pass
            N_un = Q_un.shape[2]
            attn_output_back = remerge_mapping_attn_out(attn_output, mapping_indices, N_un) # (B, num_heads, N_un, head_dim)
        else:
            attn_output_back = attn_output # (B, num_heads, 1, head_dim)

        # Rearrange back to the original hidden dimension.
        # attn_output_back: (B, num_heads, N_un, head_dim) -> (B, N_un, num_heads, head_dim)
        attn_output_back = attn_output_back.transpose(1, 2).contiguous()
        # Then reshape to (B, N_un, total_dim) and pass through output projection.
        attn_output_back = attn_output_back.reshape(*input_shape, -1)
        attn_output_back = self.o_proj(attn_output_back)
        return attn_output_back, attn_weights

class CustomLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = CustomLlamaAttention(config=config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        im_pos: Optional[torch.LongTensor] =None,
        pos_transform: Optional[torch.Tensor] = None,  # Add pos_transform as an input
        mapping_indices: Optional[torch.LongTensor] = None, # (N,) where the indice is in range [0, N_un)
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # print(im_pos, pos_transform, list(kwargs.keys()))
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            mapping_indices=mapping_indices,
            **kwargs,
        )

        hidden_states = residual + hidden_states

        # Fully Connected (MLP)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
@dataclass
class CustomBaseModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None



from transformers.utils.deprecation import deprecate_kwarg
class CustomDynamicCache(DynamicCache):
    @deprecate_kwarg("num_hidden_layers", version="4.47.0")
    def __init__(self, num_hidden_layers: Optional[int] = None) -> None:
        super().__init__()
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.cos_cache: List[torch.Tensor] = [] # same for all layers #TODO: improve memory
        self.sin_cache: List[torch.Tensor] = [] # same for all layers #TODO: improve memory
        self.mapping_indices: List[torch.Tensor] = [] # same for all layers #TODO: improve memory

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx], 
                    self.cos_cache[layer_idx], self.sin_cache[layer_idx], 
                    self.mapping_indices[layer_idx]
                )
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx], 
                   self.cos_cache[layer_idx], self.sin_cache[layer_idx],
                   self.mapping_indices[layer_idx]
                )

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if key_states is not None:
            cos = cache_kwargs.get("cos", None)
            sin = cache_kwargs.get("sin", None)
            mapping_indices = cache_kwargs.get("mapping_indices", None)
            if len(self.key_cache) <= layer_idx:
                # There may be skipped layers, fill them with empty lists
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append([])
                    self.value_cache.append([])
                    self.cos_cache.append([])
                    self.sin_cache.append([])
                    self.mapping_indices.append([])
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
                self.cos_cache.append(cos)
                self.sin_cache.append(sin)
                self.mapping_indices.append(mapping_indices)
            elif (
                len(self.key_cache[layer_idx]) == 0
            ):  # fills previously skipped layers; checking for tensor causes errors
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
                self.cos_cache[layer_idx] = cos
                self.sin_cache[layer_idx] = sin
                self.mapping_indices[layer_idx] = mapping_indices
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
                self.cos_cache[layer_idx] = torch.cat([self.cos_cache[layer_idx], cos], dim=-2)
                self.sin_cache[layer_idx] = torch.cat([self.sin_cache[layer_idx], sin], dim=-2)
                if self.mapping_indices[layer_idx] is not None and mapping_indices is not None:
                    self.mapping_indices[layer_idx] = torch.cat([self.mapping_indices[layer_idx], mapping_indices], dim=-1) # 1D tensor

        return self.key_cache[layer_idx], self.value_cache[layer_idx], self.cos_cache[layer_idx], self.sin_cache[layer_idx], self.mapping_indices[layer_idx]

    def crop(self, max_length: int):
        """Crop the past key values up to a new `max_length` in terms of tokens. `max_length` can also be
        negative to remove `max_length` tokens. This is used in assisted decoding and contrastive search."""
        # In case it is negative
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        self._seen_tokens = max_length
        for idx in range(len(self.key_cache)):
            if self.key_cache[idx] != []:
                self.key_cache[idx] = self.key_cache[idx][..., :max_length, :]
                self.value_cache[idx] = self.value_cache[idx][..., :max_length, :]
                self.cos_cache[idx] = self.cos_cache[idx][..., :max_length, :]
                self.sin_cache[idx] = self.sin_cache[idx][..., :max_length, :]
                if self.mapping_indices[idx] is not None:
                    self.mapping_indices[idx] = self.mapping_indices[idx][..., :max_length, :]


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)
        self.layers = nn.ModuleList(
            [CustomLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        pos_tracking = kwargs.pop("pos_tracking", None) # (B, N_un_img, N_img)
        pos_transform = kwargs.pop("pos_transform", None)
        im_pos = kwargs.pop("im_pos", None)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            # past_key_values = DynamicCache()
            past_key_values = CustomDynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        mapping_indices = kwargs.pop("mapping_indices", None)
        if mapping_indices is not None:
            # N_total -> N_un
            N_un = mapping_indices.max().item() + 1
            hidden_states = remerge_mapping_hidden_states(hidden_states, mapping_indices, N_un) # (B, N_un, D)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None


        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    im_pos,
                    pos_transform,
                    mapping_indices
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    im_pos=im_pos,
                    pos_transform=pos_transform,
                    mapping_indices=mapping_indices,
                    **kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = CustomBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns
        )
        return output if return_dict else output.to_tuple()



ALL_CACHE_NAMES = [
    "past_key_values",  # default
    "cache_params",  # mamba-based models
    "state",  # rwkv
    "mems",  # xlnet
    "past_buckets_states",  # reformer
]

from transformers.generation.configuration_utils import GenerationConfig

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel

class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position=None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        #
        im_pos=None,
        pos_transform=None,
        pos_tracking=None,
        mapping_indices=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels, 
                im_pos, 
                pos_transform,
                pos_tracking,
                mapping_indices
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            im_pos=im_pos,
            pos_transform=pos_transform,
            pos_tracking=pos_tracking,
            mapping_indices=mapping_indices,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        im_pos = kwargs.pop("im_pos", None)
        pos_transform = kwargs.pop("pos_transform", None)
        pos_tracking = kwargs.pop("pos_tracking", None)
        mapping_indices = kwargs.pop("mapping_indices", None)
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                im_pos, 
                pos_transform,
                pos_tracking,
                mapping_indices
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            im_pos=im_pos,
            pos_transform=pos_transform,
            pos_tracking=pos_tracking,
            mapping_indices=mapping_indices,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)

        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs
    
    def _prepare_cache_for_generation(
        self,
        generation_config: GenerationConfig,
        model_kwargs: Dict,
        assistant_model: "PreTrainedModel",
        batch_size: int,
        max_cache_length: int,
        device: torch.device,
    ) -> bool:
        """
        Prepares the cache for generation (if applicable), given `generate`'s parameterization. If a cache is
        instantiated, writes it to `model_kwargs`, under the name expected by the model.
        """
        
        cache_name = "past_key_values" if "mamba" not in self.__class__.__name__.lower() else "cache_params"

        # Quick escape route 1: if the user specifies a cache, we only need to:
        # a) check for conflicting `generate` arguments
        # b) convert to the new cache format (if the user passes a legacy cache and model supports it)
        user_defined_cache = model_kwargs.get(cache_name)
        if user_defined_cache is not None:
            raise NotImplementedError

        # Quick escape route 2: if the user specifies no cache is to be used. (conflicting arguments are handled in
        # `generation_config.validate()`)
        if generation_config.use_cache is False:
            return

        # Quick escape route 3: model that only supports legacy caches = nothing to prepare
        if not self._supports_default_dynamic_cache():
            raise NotImplementedError

        if assistant_model is not None and generation_config.cache_implementation is not None:
            logger.warning_once(
                "An assistant model is provided, using a dynamic cache instead of a cache of type="
                f"'{generation_config.cache_implementation}'."
            )
            generation_config.cache_implementation = None

        if generation_config.cache_implementation is not None:
            raise NotImplementedError
        # Use DynamicCache() instance by default. This will avoid back and forth from legacy format that
        # keeps copying the cache thus using much more memory
        else:
            model_kwargs[cache_name] = CustomDynamicCache()

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)