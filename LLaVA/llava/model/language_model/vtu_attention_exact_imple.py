
from typing import TYPE_CHECKING, List, Optional, Tuple, Union, Callable, Dict, Any

import ipdb
import torch
import torch.nn as nn
from transformers.generation.utils import GenerateOutput

from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
)
from torch_scatter import scatter_mean

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def remerge_mapping_attn_out(Y: torch.Tensor, mapping_indices: torch.Tensor, N_un: int = None) -> torch.Tensor:
    return scatter_mean(Y, mapping_indices, dim=2, dim_size=N_un)

def remerge_mapping_hidden_states(X: torch.Tensor, mapping_indices: torch.Tensor, N_un: int) -> torch.Tensor:
    return scatter_mean(X, mapping_indices, dim=1, dim_size=N_un)



from transformers.utils.deprecation import deprecate_kwarg
class CustomDynamicCache(DynamicCache):
    @deprecate_kwarg("num_hidden_layers", version="4.47.0")
    def __init__(self, num_hidden_layers: Optional[int] = None) -> None:
        super().__init__()
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.cos_cache: List[torch.Tensor] = [] 
        self.sin_cache: List[torch.Tensor] = [] 
        self.mapping_indices: List[torch.Tensor] = []

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


class CustomLlamaAttention(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[CustomDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        mapping_indices: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        # hidden_states: (B, N_un, total_dim)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        Q_un = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2) # (B, num_heads, N_un, head_dim)
        K_un = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        V_un = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings # (1, N, head_dim)

        new_mapping_index = None # None for the first forward pass
        if mapping_indices is not None: # for the first forward pass during generation
            # take the first half
            N = mapping_indices.shape[0]
            N_un = Q_un.shape[2]

            D = Q_un.shape[-1]
            assert D == cos.shape[-1] == sin.shape[-1]
            C = cos[..., :D // 2] # (1, N, D//2)
            S = sin[..., :D // 2] # (1, N, D//2)
            CM = torch.zeros(N, N_un, D // 2, device=Q_un.device, dtype=Q_un.dtype) # (N, N_un, D//2)
            SM = torch.zeros(N, N_un, D // 2, device=Q_un.device, dtype=Q_un.dtype) # (N, N_un, D//2)
            CM[torch.arange(N), mapping_indices] = C[0] # (N, N_un, D//2)
            SM[torch.arange(N), mapping_indices] = S[0] # (N, N_un, D//2)

            # q_un
            Q_un_1, Q_un_2 = torch.chunk(Q_un, 2, dim=-1)
            Q_un_reshape = torch.stack((Q_un_1, Q_un_2), dim=-2) # (B, num_heads, N_un, 2, D//2)

            # k_un
            K_un_1, K_un_2 = torch.chunk(K_un, 2, dim=-1)
            K_un_reshape = torch.stack((K_un_1, K_un_2), dim=-2) # (B, num_heads, N_un, 2, D//2)

            # k_un rotated
            K_un_rot_reshape = torch.stack((-K_un_2, K_un_1), dim=-2) # (B, num_heads, N_un, 2, D//2) # -x2, x1

            # transposed
            K_un_T_reshape = K_un_reshape.transpose(2, 3) # (B, num_heads, 2, N_un, D//2)
            K_un_rot_T_reshape = K_un_rot_reshape.transpose(2, 3) # (B, num_heads, 2, N_un, D//2)

            Q_un_K_un_T = torch.einsum('b h i r d, b h r j d -> b h i j d', Q_un_reshape, K_un_T_reshape) # (B, num_heads, N_un, N_un, D//2)
            Q_un_x_K_un_T = - torch.einsum('b h i r d, b h r j d -> b h i j d', Q_un_reshape, K_un_rot_T_reshape) # (B, num_heads, N_un, N_un, D//2)

            # compute attention scores
            term1 = torch.einsum("i r d, b h r s d, j s d -> b h i j", CM, Q_un_K_un_T, CM) # (CM) (Q_un K^T_un) (M^T C)
            term2 = torch.einsum("i r d, b h r s d, j s d -> b h i j", SM, Q_un_K_un_T, SM) # (SM) (Q_un K^T_un) (M^T S)
            term3 = torch.einsum("i r d, b h r s d, j s d -> b h i j", SM, Q_un_x_K_un_T, CM) # (SM) (Q_un x K^T_un) (M^T C)
            term4 = torch.einsum("i r d, b h r s d, j s d -> b h i j", CM, Q_un_x_K_un_T, SM) # (CM) (Q_un x K^T_un) (M^T S)
            
            ## Combine the four terms.
            attn_scores = (term1 + term2 + term3 - term4) * self.scaling

            ### Cache expanded K, V: using CustomDynamicCache
            K_m = K_un.index_select(dim=2, index=mapping_indices)
            V_m = V_un.index_select(dim=2, index=mapping_indices)
            if past_key_value is not None:
                cache_kwargs = {"sin": sin.unsqueeze(1), "cos": cos.unsqueeze(1), "mapping_indices": mapping_indices, "cache_position": cache_position}
                _,_,_,_,_ = past_key_value.update(K_m, V_m, self.layer_idx, cache_kwargs) # concat key, value and cos, sin for kv; updated mapping_indices

        else:
            Q_m = Q_un # (B, num_heads, 1, head_dim)
            K_m = K_un # (B, num_heads, 1, head_dim)
            V_m = V_un # (B, num_heads, 1, head_dim)
            C = cos.unsqueeze(1) # (1, 1, 1, head_dim)
            S = sin.unsqueeze(1) # (1, 1, 1, head_dim)

            # get previous mapping_indices
            if past_key_value is not None and len(past_key_value.mapping_indices) > self.layer_idx: # the second statement handles cases where we don't have mapping_indices from the beginning
                prev_mapping_indices = past_key_value.mapping_indices[self.layer_idx]
                if prev_mapping_indices is not None:
                    # find max
                    prev_un_len = prev_mapping_indices.max().item() + 1
                    new_mapping_index = torch.tensor([prev_un_len], device=prev_mapping_indices.device, dtype=prev_mapping_indices.dtype)
                    mapping_indices = new_mapping_index

            C_q = C # cosine for query
            S_q = S # sin for query

            ## Update keys/values with cache if using one. saving expanded key values
            if past_key_value is not None:
                cache_kwargs = {"sin": S, "cos": C, "mapping_indices": mapping_indices, "cache_position": cache_position}
                K_m, V_m, C, S, mapping_indices = past_key_value.update(K_m, V_m, self.layer_idx, cache_kwargs) # concat key, value and cos, sin for kv; updated mapping_indices

            ## Compute rotated versions for cross terms.
            K_m_rot = rotate_half(K_m) # -x2, x1

            ## Compute the four terms.
            term1 = torch.matmul(Q_m * C_q, (K_m * C).transpose(-2, -1)) # (C M) (Q_un K^T_un) (M^T C)
            term2 = torch.matmul(Q_m * S_q, (K_m * S).transpose(-2, -1)) # S M Q_un K^T_un M^T S
            term3 = torch.matmul(Q_m * S_q, (K_m_rot * C).transpose(-2, -1)) # S M Q_un x K^T_un M^T C
            term3 = -term3
            term4 = torch.matmul(Q_m * C_q, (K_m_rot * S).transpose(-2, -1)) # C M Q_un x K^T_un M^T S
            term4 = -term4

            ## Combine the four terms.
            attn_scores = (term1 + term2 + term3 - term4) * self.scaling


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
    



