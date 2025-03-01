
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
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None, pos_tracking: torch.Tensor = None # (b, s, s_ori)
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

    if pos_tracking is not None:
        pos_tracking = merge(pos_tracking, mode="sum")

    return x, size, pos_tracking

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
            j = rb if rb < len(edge_idx) else len(edge_idx) - 1
            batch_threshold = node_max[edge_idx[j, 0]]
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
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None, pos_tracking: torch.Tensor = None # (b, s, s_ori)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x, padding_mask = merge(x * size, mode="sum")
    size, _ = merge(size, mode="sum")

    if pos_tracking is not None:
        pos_tracking, _ = merge(pos_tracking, mode="sum")

    if padding_mask is not None:
        # Rearrange x, padding_mask, and size so that non-padding instances are at the front
        # padding_mask is 0 for non-padding and 1 for padding
        sort_indices = torch.argsort(padding_mask, dim=1)
        # Use gather to rearrange x, padding_mask, and size according to sort_indices
        x = x.gather(1, sort_indices.unsqueeze(-1).expand(-1, -1, x.size(2)))
        padding_mask = padding_mask.gather(1, sort_indices)
        size = size.gather(1, sort_indices.unsqueeze(-1).expand(-1, -1, size.size(2))) # (b, s, 1)
        if pos_tracking is not None:
            pos_tracking = pos_tracking.gather(1, sort_indices.unsqueeze(-1).expand(-1, -1, pos_tracking.size(2)))

    x = x / size

    # Truncate to the maximum length
    # max_len = int((1 - padding_mask).sum(dim=-1).max().item()) # this causes incorrect max_len calculation
    max_len = int((1 - padding_mask).to(torch.int64).sum(dim=-1).max().item())
    x = x[:, :max_len]
    padding_mask = padding_mask[:, :max_len]
    size = size[:, :max_len]
    if pos_tracking is not None:
        pos_tracking = pos_tracking[:, :max_len]
    return x, size, padding_mask, pos_tracking

def batch_level_merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    raise NotImplementedError("Unmerge not implemented yet.")

#####################

def repeat_merged_tokens_w_pos_tracking(merged_tokens, pos_tracking=None):
    """
    Args:
        merged_tokens (Tensor): shape (B, merged_num, hidden_size)
        pos_tracking (Tensor, optional): shape (B, merged_num, target_len)
            For example, suppose merged token num is 2 and target_len is 4;
            pos_tracking might be:
                [[1, 0, 0, 1],
                 [0, 1, 1, 0]]
            meaning token 0 should be repeated in positions 0 and 3, 
            and token 1 in positions 1 and 2.
            
    Returns:
        Tensor: shape (B, target_len, hidden_size)
            Each target position is filled with the corresponding merged token.
    """
    if pos_tracking is None:
        return merged_tokens
    else:
        # Ensure pos_tracking is of float type (in case it is provided as a boolean tensor)
        pos_tracking = pos_tracking.float()
        # Transpose pos_tracking to shape (B, target_len, merged_num)
        # Then use batch matrix multiplication to "gather" the merged tokens to the target positions
        repeated_tokens = torch.bmm(pos_tracking.transpose(1, 2), merged_tokens)
        return repeated_tokens

if __name__ == "__main__":
    # test dummy example
    bsz = 2
    seq_len = 4
    hdim = 1
    r = 2

    metric = torch.randn(bsz, seq_len, hdim)
    x = torch.randn(bsz, seq_len, hdim)
    padding_mask = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]])
    pos_tracking = torch.eye(seq_len, dtype=torch.int32).unsqueeze(0).expand(bsz, -1, -1)

    print("x:", x)
    print("metric:", metric)
    print("pos_tracking:", pos_tracking)
    
    import pdb; pdb.set_trace()
    
    ## instance level 
    # merge, unmerge = bipartite_soft_matching(metric, r, class_token=False)
    # new_x, size, pos_tracking = merge_wavg(merge, x, size = None, pos_tracking=pos_tracking)
    
    # import pdb; pdb.set_trace()
    
    ## batch level
    merge, unmerge, batch_threshold = batch_level_bipartite_soft_matching(metric, r, padding_mask=padding_mask)
    new_x, size, padding_mask, pos_tracking = batch_level_merge_wavg(merge, x, size = None, pos_tracking = pos_tracking)

    import pdb; pdb.set_trace()
    # reconstruct original length by repeating the merged tokens
    repeated_tokens = repeat_merged_tokens_w_pos_tracking(new_x, pos_tracking)
    print("repeated_tokens:", repeated_tokens)
    import pdb; pdb.set_trace()
    


