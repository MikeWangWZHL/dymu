from src.open_clip.tome import (
    bipartite_soft_matching,
    # batch_level_bipartite_soft_matching,
    merge_wavg,
    # batch_level_merge_wavg,
    do_nothing
)
from einops import rearrange

from typing import Callable, Optional, Tuple
import math
import torch

def bipartite_soft_matching_threshold(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
    specified_threshold: Optional[float] = None,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%) on an instance level.
    
    Input size is [batch, tokens, channels].
    r indicates the maximum number of tokens to remove (max 50% of tokens).
    
    Extra args:
      - class_token: Whether a class token exists (it won't be merged).
      - distill_token: Whether a distillation token exists (it won't be merged).
      - specified_threshold: If provided, only merge token pairs whose maximum matching score exceeds this threshold.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only remove at most 50% of tokens (excluding protected ones)
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
        
        # If a threshold is provided, adjust r per instance based on token scores.
        if specified_threshold is not None:
            assert metric.shape[0], "batch size > 1 not supported"
            # Count, per instance, how many token-pairs exceed the threshold.
            merge_counts = (node_max > specified_threshold).sum(dim=-1)
            # Use the minimum count across the batch as the effective number to merge.
            r_effective = int(merge_counts[0].item())
            # r = min(r, r_effective)
            r = min(r_effective, (t - protected) // 2)

            if r <= 0:
                return do_nothing, do_nothing

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
            # padding_mask is 0 for non-padding and 1 for padding
            mask_a = padding_mask[..., ::2].unsqueeze(2).bool()  # Shape: (b, s//2, 1)
            mask_b = padding_mask[..., 1::2].unsqueeze(1).bool()  # Shape: (b, 1, s//2)

            # Combine masks to identify where either 'a' or 'b' has padding
            combined_mask = mask_a | mask_b  # Shape: (b, s//2, s//2)

            # Set scores at padding positions to -inf
            # scores = scores.masked_fill(combined_mask, -math.inf)
            scores.masked_fill_(combined_mask, -math.inf)

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
            # rb = sum(node_max > specified_threshold) # merge all tokens over the specified_threshold
            rb = int((node_max > specified_threshold).sum().item())
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
        
        # 
        # dst_new = dst_flat.clone()
        dst_new = dst_flat
        dst_new = update_dst(dst_new, src_tokens, index_expanded, reduce=mode, include_self=True)
        dst_new = dst_new.reshape(dst.size())
        # print("dst tokens merged:", dst_new)

        # # construct new x
        # x_new = x.clone()
        x_new = x
        x_new[..., :src.size(1), :] = src
        x_new[..., src.size(1):, :] = dst_new
        # x_new[src_b, src_s, :] = torch.zeros_like(src[src_b, src_s, :])
        x_new[src_b, src_s, :] = 0

        if padding_mask is not None:
            # padding_mask_src, padding_mask_dst = padding_mask[..., ::2].clone(), padding_mask[..., 1::2].clone()
            padding_mask_src, padding_mask_dst = padding_mask[..., ::2], padding_mask[..., 1::2]
            padding_mask_src[src_b, src_s] = 1
            new_padding_mask = torch.concatenate((padding_mask_src, padding_mask_dst), dim=1)
        else:
            new_padding_mask = torch.zeros((bsz, seq_len), device=x.device, dtype=x.dtype)
            # new_padding_mask_a = new_padding_mask[..., ::2].clone()
            new_padding_mask_a = new_padding_mask[..., ::2]
            new_padding_mask_a[src_b, src_s] = 1
            new_padding_mask[..., :src.size(1)] = new_padding_mask_a
        
        # # construct padding masking: (b, s); 0 for non-padding, 1 for padding; fill in 1 where x_new is zero
        # padding_mask = torch.all(x_new == 0, dim=-1).int().to(x.device)
        return x_new, new_padding_mask


    def unmerge(x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Unmerge not implemented yet.")

    return merge, unmerge, batch_threshold

def batch_level_merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None, pos_tracking: torch.Tensor = None, # (b, s, s_ori)
    cls_token: bool = False
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

    assert padding_mask is not None
    if cls_token:
        if x.size(1) > 1:
            # Separate the cls token (first token)
            cls_token_x = x[:, :1, :]
            cls_token_padding = padding_mask[:, :1]
            cls_token_size = size[:, :1, :]
            if pos_tracking is not None:
                cls_token_pos = pos_tracking[:, :1, :]
            # Process the rest of the tokens (from index 1 onward)
            rest_x = x[:, 1:]
            rest_padding_mask = padding_mask[:, 1:]
            rest_size = size[:, 1:]
            if pos_tracking is not None:
                rest_pos_tracking = pos_tracking[:, 1:]
            # Sort only the rest tokens based on the padding mask
            sort_indices = torch.argsort(rest_padding_mask, dim=1)
            rest_x = rest_x.gather(1, sort_indices.unsqueeze(-1).expand(-1, -1, x.size(2)))
            rest_padding_mask = rest_padding_mask.gather(1, sort_indices)
            rest_size = rest_size.gather(1, sort_indices.unsqueeze(-1).expand(-1, -1, size.size(2)))
            if pos_tracking is not None:
                rest_pos_tracking = rest_pos_tracking.gather(
                    1, sort_indices.unsqueeze(-1).expand(-1, -1, pos_tracking.size(2))
                )
            # Recombine the unchanged cls token with the sorted tokens
            x = torch.cat([cls_token_x, rest_x], dim=1)
            padding_mask = torch.cat([cls_token_padding, rest_padding_mask], dim=1)
            size = torch.cat([cls_token_size, rest_size], dim=1)
            if pos_tracking is not None:
                pos_tracking = torch.cat([cls_token_pos, rest_pos_tracking], dim=1)
        else:
            # if there is only one token, do nothing
            pass
    else:
        # Rearrange x, padding_mask, and size so that non-padding instances are at the front
        # padding_mask is 0 for non-padding and 1 for padding
        sort_indices = torch.argsort(padding_mask, dim=1)
        # Use gather to rearrange x, padding_mask, and size according to sort_indices
        x = x.gather(1, sort_indices.unsqueeze(-1).expand(-1, -1, x.size(2)))
        padding_mask = padding_mask.gather(1, sort_indices)
        size = size.gather(1, sort_indices.unsqueeze(-1).expand(-1, -1, size.size(2))) # (b, s, 1)
        if pos_tracking is not None:
            pos_tracking = pos_tracking.gather(1, sort_indices.unsqueeze(-1).expand(-1, -1, pos_tracking.size(2)))

    x = x / (size+1e-4)

    # Truncate to the maximum length
    max_len = int((padding_mask < 0.5).to(torch.int64).sum(dim=-1).max().item()) # 0 for non-padding, 1 for padding

    x = x[:, :max_len]
    padding_mask = padding_mask[:, :max_len]
    size = size[:, :max_len]
    if pos_tracking is not None:
        pos_tracking = pos_tracking[:, :max_len]
    return x, size, padding_mask, pos_tracking




### this is worse
from torch.nn import functional as F

@torch.jit.script
def bipartite_soft_matching_script(
    metric: torch.Tensor,           # (b, s, c)
    r: int,
    class_token: bool,
    distill_token: bool,
    padding_mask: Optional[torch.Tensor],  # (b, s)
    max_r_per_instance: int,
    specified_threshold: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes matching indices for bipartite soft matching.
    Returns:
      unm_idx: Tensor of unmerged token indices
      src_idx: Tensor of merged token source indices
      dst_idx: Tensor of merged token destination indices
      batch_threshold: A tensor holding the threshold for the batch
    """
    protected: int = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    bsz: int = metric.size(0)
    seq_len: int = metric.size(1)
    r = min(r, (seq_len - protected) // 2)
    if r <= 0:
        return (torch.empty((0, 1), dtype=torch.long, device=metric.device),
                torch.empty((0, 1), dtype=torch.long, device=metric.device),
                torch.empty(0, dtype=torch.long, device=metric.device),
                torch.tensor(0.0, device=metric.device))
    
    # Normalize using F.normalize for stability.
    metric = F.normalize(metric, p=2.0, dim=-1)
    # Split tokens into two groups (even-indexed and odd-indexed)
    a = metric[..., ::2, :]
    b_tokens = metric[..., 1::2, :]
    scores = torch.matmul(a, b_tokens.transpose(-1, -2))  # (b, s//2, s//2)

    if class_token:
        scores[..., 0, :] = -math.inf
    if distill_token:
        scores[..., :, 0] = -math.inf

    if padding_mask is not None:
        # Build boolean masks for groups a and b.
        mask_a = padding_mask[..., ::2].unsqueeze(2).to(torch.bool)
        mask_b = padding_mask[..., 1::2].unsqueeze(1).to(torch.bool)
        combined_mask = mask_a | mask_b
        scores = scores.masked_fill(combined_mask, -math.inf)

    if max_r_per_instance > 0:
        node_max_instance, _ = torch.max(scores, -1)
        edge_idx_instance = torch.argsort(node_max_instance, dim=-1, descending=True).unsqueeze(-1)
        unm_idx_instance = edge_idx_instance[..., max_r_per_instance:, :]
        unm_idx_instance_expanded = unm_idx_instance.expand(-1, -1, scores.size(-1))
        batch_indices = torch.arange(bsz, device=scores.device).view(-1, 1, 1).expand_as(unm_idx_instance_expanded)
        scores[batch_indices, unm_idx_instance_expanded, :] = -math.inf

    # Flatten scores for a global matching over the batch.
    # scores = rearrange(scores, 'b i j -> (b i) j')
    scores = scores.reshape(-1, scores.size(-1))
    node_max, node_idx = torch.max(scores, -1)
    edge_idx = torch.argsort(node_max, descending=True).unsqueeze(-1)

    rb: int = 0
    if specified_threshold > 0:
        rb = int(torch.sum(node_max > specified_threshold).item())
    else:
        rb = r * bsz

    unm_idx = edge_idx[rb:]
    src_idx = edge_idx[:rb]
    # Gather destination indices corresponding to the best match.
    dst_idx = node_idx.index_select(0, src_idx.squeeze())

    batch_threshold: torch.Tensor = torch.tensor(0.0, device=metric.device)
    if specified_threshold > 0:
        batch_threshold = torch.tensor(0.0, device=metric.device)
    else:
        j: int = rb if rb < edge_idx.size(0) else edge_idx.size(0) - 1
        batch_threshold = node_max[edge_idx[j, 0]]
        batch_threshold = torch.max(batch_threshold, torch.tensor(0.0, device=metric.device))
    
    return unm_idx, src_idx, dst_idx, batch_threshold

@torch.jit.script
def update_dst_script(
    dst_flat: torch.Tensor,
    src_elements: torch.Tensor,
    index_expanded: torch.Tensor,
    mode: str,
    include_self: bool
) -> torch.Tensor:
    """
    Performs the in-place update of destination tokens.
    """
    if mode == "sum":
        dst_flat.scatter_add_(0, index_expanded, src_elements)
        return dst_flat
    elif mode == "mean":
        counts = torch.zeros_like(dst_flat)
        ones = torch.ones_like(src_elements)
        sum_dst_flat = torch.zeros_like(dst_flat)
        sum_dst_flat.scatter_add_(0, index_expanded, src_elements)
        counts.scatter_add_(0, index_expanded, ones)
        if include_self:
            sum_dst_flat += dst_flat
            counts += (dst_flat != 0).to(counts.dtype)
        counts = counts.clamp(min=1)
        return sum_dst_flat / counts
    else:
        return dst_flat

@torch.jit.script
def merge_tokens_script(
    x: torch.Tensor,
    unm_idx: torch.Tensor,
    src_idx: torch.Tensor,
    dst_idx: torch.Tensor,
    padding_mask: Optional[torch.Tensor],
    mode: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Merges tokens based on provided matching indices.
    Returns the merged tensor and updated padding mask.
    """
    src = x[..., ::2, :]
    dst = x[..., 1::2, :]
    t1: int = src.size(1)
    src_idx_squeezed = src_idx.squeeze(-1)
    src_b = src_idx_squeezed // t1
    src_s = src_idx_squeezed % t1
    src_tokens = src[src_b, src_s, :]
    dst_seq_len: int = dst.size(1)
    dst_flat_indices = src_b * dst_seq_len + dst_idx
    dst_flat = dst.reshape(-1, dst.size(-1))
    index_expanded = dst_flat_indices.unsqueeze(-1).expand(-1, dst_flat.size(-1))
    dst_new = dst_flat
    dst_new = update_dst_script(dst_new, src_tokens, index_expanded, mode, True)
    dst_new = dst_new.reshape(dst.size())
    x_new = x
    x_new[..., :src.size(1), :] = src
    x_new[..., src.size(1):, :] = dst_new
    # Zero-out the merged source tokens.
    x_new[src_b, src_s, :] = 0

    if padding_mask is not None:
        padding_mask_src = padding_mask[..., ::2]
        padding_mask_dst = padding_mask[..., 1::2]
        padding_mask_src[src_b, src_s] = 1
        new_padding_mask = torch.cat((padding_mask_src, padding_mask_dst), 1)
    else:
        bsz: int = x.size(0)
        seq_len: int = x.size(1)
        new_padding_mask = torch.zeros((bsz, seq_len), device=x.device, dtype=x.dtype)
        new_padding_mask_a = new_padding_mask[..., ::2]
        new_padding_mask_a[src_b, src_s] = 1
        new_padding_mask[..., :src.size(1)] = new_padding_mask_a

    return x_new, new_padding_mask

@torch.jit.script
def batch_level_merge_wavg_script(
    x: torch.Tensor,
    size: Optional[torch.Tensor],
    pos_tracking: torch.Tensor,
    padding_mask: torch.Tensor,
    cls_token: bool,
    unm_idx: torch.Tensor,
    src_idx: torch.Tensor,
    dst_idx: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Applies the merge operation with a weighted average.
    Expects the matching indices computed by bipartite_soft_matching_script.
    """
    if size is None:
        size = torch.ones_like(x[..., :1])
    
    # Merge weighted sum.
    x, padding_mask = merge_tokens_script(x * size, unm_idx, src_idx, dst_idx, padding_mask, "sum")
    size, _ = merge_tokens_script(size, unm_idx, src_idx, dst_idx, padding_mask, "sum")

    pos_tracking, _ = merge_tokens_script(pos_tracking, unm_idx, src_idx, dst_idx, padding_mask, "sum")
    
    if cls_token:
        if x.size(1) > 1:
            cls_token_x = x[:, :1, :]
            cls_token_padding = padding_mask[:, :1]
            cls_token_size = size[:, :1, :]
            cls_token_pos = pos_tracking[:, :1, :]

            rest_x = x[:, 1:]
            rest_padding = padding_mask[:, 1:]
            rest_size = size[:, 1:]
            rest_pos = pos_tracking[:, 1:]

            sort_indices = torch.argsort(rest_padding, dim=1)
            rest_x = rest_x.gather(1, sort_indices.unsqueeze(-1).expand(-1, -1, x.size(2)))
            rest_padding = rest_padding.gather(1, sort_indices)
            rest_size = rest_size.gather(1, sort_indices.unsqueeze(-1).expand(-1, -1, size.size(2)))
            rest_pos = rest_pos.gather(1, sort_indices.unsqueeze(-1).expand(-1, -1, pos_tracking.size(2)))

            x = torch.cat([cls_token_x, rest_x], 1)
            padding_mask = torch.cat([cls_token_padding, rest_padding], 1)
            size = torch.cat([cls_token_size, rest_size], 1)
            pos_tracking = torch.cat([cls_token_pos, rest_pos], 1)
    else:
        sort_indices = torch.argsort(padding_mask, dim=1)
        x = x.gather(1, sort_indices.unsqueeze(-1).expand(-1, -1, x.size(2)))
        padding_mask = padding_mask.gather(1, sort_indices)
        size = size.gather(1, sort_indices.unsqueeze(-1).expand(-1, -1, size.size(2)))
        pos_tracking = pos_tracking.gather(1, sort_indices.unsqueeze(-1).expand(-1, -1, pos_tracking.size(2)))
    
    x = x / (size + 1e-4)
    max_len = int((padding_mask < 0.5).to(torch.int64).sum(dim=1).max().item())
    x = x[:, :max_len]
    padding_mask = padding_mask[:, :max_len]
    size = size[:, :max_len]
    pos_tracking = pos_tracking[:, :max_len]
    
    return x, size, padding_mask, pos_tracking



def merge_tokens(merge_mode, metric, r, hidden_states, size=None, padding_mask=None, pos_tracking=None, threshold=None):
    if merge_mode == "instance_level":
        merge, _ = bipartite_soft_matching(
            metric,
            r,
            True,
            False,
        )
        hidden_states, size, pos_tracking = merge_wavg(
            merge, hidden_states, size, pos_tracking=pos_tracking
        )
    elif merge_mode == "batch_level":
        specified_threshold = threshold
        # B = hidden_states.shape[0]
        # if B == 1:

        #     merge, _, = bipartite_soft_matching_threshold(
        #         metric,
        #         r,
        #         True,
        #         False,
        #         specified_threshold=specified_threshold
        #     )
        #     if merge != do_nothing:

        #         hidden_states, size, pos_tracking = merge_wavg(
        #             merge, hidden_states, size, pos_tracking=pos_tracking
        #         )
        # else:

        # original
        merge, _, batch_threshold = batch_level_bipartite_soft_matching(
            metric,
            r,
            True,
            False,
            padding_mask = padding_mask,
            max_r_per_instance = None,
            specified_threshold = specified_threshold
        )
        if merge != do_nothing:
            hidden_states, size, padding_mask, pos_tracking = batch_level_merge_wavg(
                merge, hidden_states, size, pos_tracking=pos_tracking, cls_token=True
            )

        # ## Use scripted functions for performance
        # specified_threshold = threshold if threshold is not None else -1
        # max_r_per_instance = -1
        # unm_idx, src_idx, dst_idx, batch_threshold = bipartite_soft_matching_script(
        #     metric,
        #     r,
        #     True,   # class_token flag
        #     False,  # distill_token flag
        #     padding_mask,
        #     max_r_per_instance,      # <= 0 means no limit
        #     specified_threshold      # <= 0 means no specified threshold
        # )
        # # Check if any merging is scheduled (by verifying if src_idx is non-empty)
        # if src_idx.numel() > 0:
        #     hidden_states, size, padding_mask, pos_tracking = batch_level_merge_wavg_script(
        #         hidden_states,
        #         size,
        #         pos_tracking,
        #         padding_mask,
        #         True,     # cls_token flag
        #         unm_idx,
        #         src_idx,
        #         dst_idx
        #     )

    return hidden_states, padding_mask, pos_tracking

import torch
import time
from tqdm import tqdm
# Assuming the functions have been imported from src.open_clip.tome:
# from src.open_clip.tome import bipartite_soft_matching, batch_level_bipartite_soft_matching, merge_wavg, batch_level_merge_wavg, do_nothing

# Create dummy data
batch_size = 1
seq_len = 576
hidden_dim = 768

ts_instance_level = []
ts_batch_level = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for _ in tqdm(range(200)):
    # Dummy hidden states: (batch_size, seq_len, hidden_dim)
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim).to(device)
    # Dummy size: a list indicating the sequence length for each instance in the batch
    size = torch.ones_like(hidden_states[..., 0, None])
    # Dummy metric: for example, a similarity matrix (batch_size, seq_len, seq_len)
    metric = torch.randn(batch_size, seq_len, hidden_dim//8).to(device)
    # Dummy r: a float parameter, can represent a threshold or ratio
    r = int(seq_len * 0.2)  # 20% of seq_len
    # Dummy padding mask: (batch_size, seq_len) boolean mask (False means not padded)
    padding_mask = torch.zeros(batch_size, seq_len).to(device)
    # Dummy position tracking: simple position indices repeated for each batch instance
    pos_tracking = torch.eye(seq_len, dtype=torch.int32).unsqueeze(0).expand(batch_size, -1, -1).to(device)

    # get dummy threshold
    metric = metric / metric.norm(dim=-1, keepdim=True)
    a, b = metric[..., ::2, :], metric[..., 1::2, :]
    scores = a @ b.transpose(-1, -2)  # shape: (b, s//2, s//2)
    scores_flat = scores.reshape(-1, scores.size(-1))  # shape: (b*(s//2), s//2)
    node_max, node_idx = scores_flat.max(dim=-1)  # (b*(s//2),)

    node_max = sorted(node_max, reverse=True)
    dummy_threshold = node_max[r*batch_size].item() # merge 0.2 tokens

    # Measure instance-level merge time
    start_time = time.time()
    _ = merge_tokens("instance_level", metric, r, hidden_states, size=size, pos_tracking=pos_tracking)
    instance_level_time = time.time() - start_time
    ts_instance_level.append(instance_level_time)

    # Measure batch-level merge time
    start_time = time.time()
    _ = merge_tokens("batch_level", metric, r, hidden_states, size=size, padding_mask=padding_mask, pos_tracking=pos_tracking, threshold=dummy_threshold)
    batch_level_time = time.time() - start_time
    ts_batch_level.append(batch_level_time)


instance_level_time = sum(ts_instance_level[1:]) / len(ts_instance_level[1:])
batch_level_time = sum(ts_batch_level[1:]) / len(ts_batch_level[1:])

print(f"Instance level time: {instance_level_time*1000:.4f} ms")
print(f"Batch level time: {batch_level_time*1000:.4f} ms")