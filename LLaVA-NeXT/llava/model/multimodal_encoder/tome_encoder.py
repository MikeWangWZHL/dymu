from typing import Optional, Tuple, Union, Dict
from dataclasses import dataclass
from functools import partial, reduce
from PIL import Image
import torch
import torch.utils.checkpoint
from torch import nn
import os
from transformers.image_processing_utils import BatchFeature, get_size_dict
from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    to_numpy_array,
)
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig
from transformers.utils import ModelOutput
from llava.utils import rank0_print

from transformers.utils import (
    is_flash_attn_2_available,
)

if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func
    from flash_attn.layers.rotary import apply_rotary_emb

else:
    flash_attn_varlen_func = None
    apply_rotary_emb = None

if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
else:
    flash_attn_varlen_func = None

class SigLipImageProcessor:
    def __init__(self, image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5), size=(384, 384), crop_size: Dict[str, int] = None, resample=PILImageResampling.BICUBIC, rescale_factor=1 / 255, data_format=ChannelDimension.FIRST):
        crop_size = crop_size if crop_size is not None else {"height": 384, "width": 384}
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")

        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format
        self.crop_size = crop_size

    def preprocess(self, images, return_tensors):
        if isinstance(images, Image.Image):
            images = [images]
        else:
            # to adapt video data
            images = [to_numpy_array(image) for image in images]
            assert isinstance(images, list)

        transforms = [
            convert_to_rgb,
            to_numpy_array,
            partial(resize, size=self.size, resample=self.resample, data_format=self.data_format),
            partial(rescale, scale=self.rescale_factor, data_format=self.data_format),
            partial(normalize, mean=self.image_mean, std=self.image_std, data_format=self.data_format),
            partial(to_channel_dimension_format, channel_dim=self.data_format, input_channel_dim=self.data_format),
        ]

        images = reduce(lambda x, f: [*map(f, x)], transforms, images)
        data = {"pixel_values": images}

        return BatchFeature(data=data, tensor_type=return_tensors)

from typing import List, Union
class SigLipVisionConfigTome(PretrainedConfig):
    model_type = "siglip_vision_model"

    def __init__(
        self,
        hidden_size=1152,
        image_mean=(0.5, 0.5, 0.5),
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_channels=3,
        image_size=384,
        patch_size=14,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        # tome args
        merge_mode: str = "batch_level", # merge mode: instance_level or batch_level
        r_total: int = 0, # total number of tokens to remove
        r_schedule: str = "constant", # r schedule: constant, linear, reverse_linear
        max_r_per_instance_ratio: float = None, # 1.0 => refer to fixed r for each instance; > 1.0 => dynamic r
        update_threshold: bool = False, # whether to post-hoc update threshold after training
        specified_thresholds: List[float] = None, # specified threshold for each layer
        set_training_mode: bool = False,
        # other settings
        repeat_merged_tokens: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.image_mean = image_mean
        self.merge_mode = merge_mode
        self.r_total = r_total
        self.r_schedule = r_schedule
        self.max_r_per_instance_ratio = max_r_per_instance_ratio
        self.update_threshold = update_threshold
        self.specified_thresholds = specified_thresholds
        self.set_training_mode = set_training_mode
        self.repeat_merged_tokens = repeat_merged_tokens
        

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from SigLipConfig
        if config_dict.get("model_type") == "siglip":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            print(f"You are using a model of type {config_dict['model_type']} to instantiate a model of type " f"{cls.model_type}. This is not supported for all configurations of models and can yield errors.")

        return cls.from_dict(config_dict, **kwargs)


class SigLipVisionEmbeddings(nn.Module):
    def __init__(self, config: SigLipVisionConfigTome):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class SigLipAttentionTome(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Copied from transformers.models.clip.modeling_clip.CLIPAttention.__init__
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:" f" {self.num_heads}).")
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        size: Optional[torch.Tensor] = None, # ToMe token size vector
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2) # (b, h, q_len, h_d)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        ### Tome ###
        B, N = batch_size, q_len
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
                full_bias = attention_mask
            else:
                full_bias = full_bias + attention_mask
        metric = key_states.mean(1) # (b, l, head_d)
        ###

        k_v_seq_len = key_states.shape[-2]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        if attn_weights.size() != (batch_size, self.num_heads, q_len, k_v_seq_len):
            raise ValueError(f"Attention weights should be of size {(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is" f" {attn_weights.size()}")

        if full_bias is not None:
            if full_bias.size() != (batch_size, 1, q_len, k_v_seq_len) and full_bias.size() != (batch_size, self.num_heads, q_len, k_v_seq_len):
                raise ValueError(f"Bias should be of size {(batch_size, 1, q_len, k_v_seq_len)}, or {(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is {full_bias.size()}")
            attn_weights = attn_weights + full_bias

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is" f" {attn_output.size()}")

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights, metric


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->SigLip
class SigLipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

import torch.distributed as dist
from open_clip.tome import (
    do_nothing,
    bipartite_soft_matching,
    merge_wavg,
    merge_source,
    batch_level_bipartite_soft_matching,
    batch_level_merge_wavg,
    batch_level_merge_source
)
# Copied from transformers.models.clip.modeling_clip.CLIPEncoderLayer with CLIP->SigLip
class SigLipEncoderLayerTome(nn.Module):
    def __init__(
        self, 
        config: SigLipVisionConfigTome, 
        # TOME args
        trace_source: bool = False,
        prop_attn: bool = True,
        cls_token: bool = False,
        r: int = 0,
        specified_threshold: float = None,
    ):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SigLipAttentionTome(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SigLipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        # ToMe configs
        self._tome_info = {
            "r": r, # number of tokens to remove
            "size": None,
            "source": None,
            "trace_source": trace_source,
            "prop_attn": prop_attn,
            "class_token": cls_token,
            "distill_token": False,
            "merge_mode": config.merge_mode,
            "max_r_per_instance_ratio": config.max_r_per_instance_ratio,
        }
        if config.max_r_per_instance_ratio is not None:
            print("setting max r per instance to: ", int(self._tome_info["max_r_per_instance_ratio"] * r))

        self.update_threshold = config.update_threshold
        self.register_buffer('threshold', torch.tensor(1.0)) # default to be no merging
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

    def merge_tokens(self, metric, r, hidden_states, padding_mask=None, pos_tracking=None):
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
            hidden_states, self._tome_info["size"], pos_tracking = merge_wavg(
                merge, hidden_states, self._tome_info["size"], pos_tracking=pos_tracking
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

            B = hidden_states.shape[0]
            if specified_threshold is not None and B == 1:
                # inference time; use efficient instance-level with threshold version
                merge, _ = bipartite_soft_matching(
                    metric,
                    r,
                    self._tome_info["class_token"],
                    self._tome_info["distill_token"],
                    specified_threshold=specified_threshold,
                    max_r_per_instance=max_r_per_instance
                )
                hidden_states, self._tome_info["size"], pos_tracking = merge_wavg(
                    merge, hidden_states, self._tome_info["size"], pos_tracking=pos_tracking
                )
            else:
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
                    hidden_states, self._tome_info["size"], padding_mask, pos_tracking = batch_level_merge_wavg(
                        merge, hidden_states, self._tome_info["size"], pos_tracking=pos_tracking, cls_token=self._tome_info["class_token"]
                    )
                    if self.training or self.update_threshold:
                        self.threshold_running_avg(batch_threshold)
            
        return hidden_states, padding_mask, pos_tracking

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

    # Ignore copy
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
        padding_mask: Optional[torch.Tensor] = None,
        pos_tracking: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)

        ### tome ###
        dtype = hidden_states.dtype
        if padding_mask is not None:
            attention_mask = self._get_attn_mask_from_padding_mask(padding_mask, dtype)
        else:
            attention_mask = None
        
        ## ToMe proportional attention
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        
        hidden_states, attn_weights, metric = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            size=attn_size,
        )
        hidden_states = residual + hidden_states

        r = self._tome_info["r"]
        if r > 0:
            hidden_states, padding_mask, pos_tracking = self.merge_tokens(metric, r, hidden_states, padding_mask=padding_mask, pos_tracking=pos_tracking)

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        final_outputs = {
            "hidden_states": outputs[0],
            "attentions": outputs[1] if output_attentions else None,
            "padding_mask": padding_mask,
            "pos_tracking": pos_tracking
        }
        return final_outputs


class SigLipPreTrainedModelTome(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SigLipVisionConfigTome
    base_model_prefix = "siglip"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        pass

@dataclass
class CLIPVisionEncoderToMEOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    sizes: Optional[Tuple[torch.FloatTensor, ...]] = None
    padding_masks: Optional[Tuple[torch.FloatTensor, ...]] = None
    pos_trackings: Optional[Tuple[torch.IntTensor, ...]] = None

# Copied from transformers.models.clip.modeling_clip.CLIPEncoder with CLIP->SigLip
class SigLipEncoderTome(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`SigLipEncoderLayerTome`].

    Args:
        config: SigLipVisionConfig
    """

    def __init__(self, config: SigLipVisionConfigTome):
        super().__init__()
        self.config = config

        ### ToMe configs ###
        self.rs = self._get_rs(
            config.num_hidden_layers, 
            config.r_total, 
            config.r_schedule
        )
        print("set total avg remove token nums each layer as: ", self.rs)
        print("merge mode: ", config.merge_mode)
        self._tome_info = {
            "size": None,
            "source": None,
            "trace_source": False,
            "prop_attn": True,
            "class_token": False,
            "distill_token": False,
            "merge_mode": config.merge_mode,
            "max_r_per_instance_ratio": config.max_r_per_instance_ratio,
            "update_threshold": config.update_threshold,
            "specified_thresholds": config.specified_thresholds,
        }
        ###

        self.layers = nn.ModuleList([
            SigLipEncoderLayerTome(
                config,
                trace_source = False,
                prop_attn = True,
                cls_token = False,
                r = self.rs[i],
                specified_threshold = config.specified_thresholds[i] if config.specified_thresholds is not None else None
            ) for i in range(config.num_hidden_layers)
        ])
        self.gradient_checkpointing = False

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

    # Ignore copy
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        self._tome_info["size"] = None
        self._tome_info["source"] = None

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        
        padding_mask = None
        padding_masks = []
        pos_trackings = []
        sizes = []
        self.output_stats = {}
        B, N = hidden_states.shape[:2]
        pos_tracking = torch.eye(N, dtype=torch.int32, device=hidden_states.device).unsqueeze(0).expand(B, -1, -1)
        for idx, encoder_layer in enumerate(self.layers):
            encoder_layer._tome_info["size"] = self._tome_info["size"]
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                    padding_mask,
                    pos_tracking
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                    padding_mask=padding_mask,
                    pos_tracking=pos_tracking
                )

            # hidden_states = layer_outputs[0]
            hidden_states = layer_outputs["hidden_states"]
            padding_mask = layer_outputs["padding_mask"]
            pos_tracking = layer_outputs["pos_tracking"]
            self._tome_info["size"] = encoder_layer._tome_info["size"]
            padding_masks.append(padding_mask)
            sizes.append(self._tome_info["size"])
            pos_trackings.append(pos_tracking)
            
            # track number of tokens after each block
            if padding_mask is not None:
                ntoks = (padding_mask<0.5).float().sum(-1)
                ntoks = ntoks.detach().tolist()
            else:
                ntoks = [hidden_states.shape[1]]*hidden_states.shape[0]
            self.output_stats[f"block_{idx}_ntoks"] = ntoks

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        # if not return_dict:
        #     return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        # return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)
        return CLIPVisionEncoderToMEOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
            sizes=sizes,
            padding_masks=padding_masks,
            pos_trackings=pos_trackings
        )



class SigLipVisionTransformer(nn.Module):
    def __init__(self, config: SigLipVisionConfigTome):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SigLipVisionEmbeddings(config)
        self.encoder = SigLipEncoderTome(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.head = SigLipMultiheadAttentionPoolingHeadTome(config)

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLIPVisionEncoderToMEOutput]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooled_output = self.head(last_hidden_state)

        # if not return_dict:
        #     return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return CLIPVisionEncoderToMEOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            padding_masks=encoder_outputs.padding_masks,
            sizes=encoder_outputs.sizes,
            pos_trackings=encoder_outputs.pos_trackings
        )


class SigLipMultiheadAttentionPoolingHeadTome(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(self, config: SigLipVisionConfigTome):
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = torch.nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SigLipMLP(config)

    def forward(self, hidden_state, attention_mask=None, size=None):
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)


        ### tome ###
        B, N = batch_size, hidden_state.shape[1]
        full_bias = None
        if size is not None:
            # replace all 0s with 1s
            size = torch.where(size < 0.5, torch.ones_like(size), size)
            size_bias_log = size.log()[:, :, 0] # (b, src_len, 1) -> (b, src_len)
            size_bias_log = size_bias_log.unsqueeze(1).unsqueeze(1).expand(B, self.attention.num_heads, 1, N) # (b, src_len) -> (b, num_heads, 1, src_len)
            full_bias = size_bias_log
        
        # apply attention mask before softmax (-inf for masked tokens)
        if attention_mask is not None:
            if attention_mask.size() != (B, 1, 1, N):
                raise ValueError(
                    f"Attention mask should be of size {(B, 1, 1, N)}, but is {attention_mask.size()}"
                )
            # Expand the attn_mask to [B, num_heads, N, N]
            expanded_mask = attention_mask.expand(B, self.attention.num_heads, 1, N)
            if full_bias is None:
                full_bias = expanded_mask
            else:
                full_bias = full_bias + expanded_mask
        
        if full_bias is not None:
            full_bias = full_bias.reshape(B * self.attention.num_heads, 1, N)  
        ###

        hidden_state = self.attention(probe, hidden_state, hidden_state, attn_mask=full_bias)[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]


class SigLipVisionModelTome(SigLipPreTrainedModelTome):
    config_class = SigLipVisionConfigTome
    main_input_name = "pixel_values"
    _no_split_modules = ["SigLipEncoderLayerTome"]

    def __init__(self, config: SigLipVisionConfigTome):
        super().__init__(config)

        self.vision_model = SigLipVisionTransformer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLIPVisionEncoderToMEOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, SigLipVisionModel

        >>> model = SigLipVisionModel.from_pretrained("google/siglip-base-patch16-224")
        >>> processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled features
        ```"""
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )


TOME_KEYS = [
    "merge_mode",
    "r_total",
    "r_schedule",
    "max_r_per_instance_ratio",
    "update_threshold",
    "specified_thresholds",
    "set_training_mode",
    "repeat_merged_tokens",
]
class TomeVisionTower(nn.Module):
    def __init__(self, vision_tower, vision_tower_cfg, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.config = SigLipVisionConfigTome()

        for key in TOME_KEYS:
            if hasattr(vision_tower_cfg, key):
                setattr(self.config, key, getattr(vision_tower_cfg, key))

        print("Loading TomeVisionTower with config: ", self.config)
        
        self.repeat_merged_tokens = getattr(self.config, 'repeat_merged_tokens', False)
        print(f"If reconstruct original sequence with repeating merged tokens: {self.repeat_merged_tokens}")

        self.vision_tower_name = vision_tower

        self.image_processor = SigLipImageProcessor()

        if not delay_load:
            rank0_print(f"Loading vision tower: {vision_tower}")
            self.load_model()
        elif getattr(vision_tower_cfg, "unfreeze_mm_vision_tower", False):
            # TODO: better detector is needed.
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()
        elif hasattr(vision_tower_cfg, "mm_tunable_parts") and "mm_vision_tower" in vision_tower_cfg.mm_tunable_parts:
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
            self.load_model()
        else:
            self.cfg_only = self.config

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
            return
        if not os.path.exists(self.vision_tower_name):
            loading_name = "google/siglip-so400m-patch14-384"
        else:
            loading_name = self.vision_tower_name
        print(f"Loading vision tower: {loading_name} with SigLipVisionModelTome class.")
        
        # self.vision_tower = SigLipVisionModelTome.from_pretrained(loading_name, device_map=device_map)
        
        # Step 1: Initialize model with updated self.config
        self.vision_tower = SigLipVisionModelTome(self.config)  # Directly pass self.config

        # Step 2: Load pretrained weights from checkpoint
        state_dict = SigLipVisionModelTome.from_pretrained(loading_name, device_map=device_map).state_dict()
        
        # Step 3: Load the state_dict into the initialized model
        missing_keys, unexpected_keys = self.vision_tower.load_state_dict(state_dict, strict=False)  # `strict=False` allows partial mismatches
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")

        if self.config.set_training_mode:
            self.vision_tower.train()
        else:
            self.vision_tower.eval()

        del self.vision_tower.vision_model.encoder.layers[-1:]
        self.vision_tower.vision_model.head = nn.Identity()
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def forward(self, images):

        if self.config.set_training_mode and not self.vision_tower.training:
            self.vision_tower.train()

        if type(images) is list:
            image_features = []
            padding_masks = []
            sizes = []
            pos_trackings = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = image_forward_out.hidden_states[-1].to(image.dtype)
                padding_mask = image_forward_out.padding_masks[-1]
                size = image_forward_out.sizes[-1]
                pos_tracking = image_forward_out.pos_trackings[-1]
                # assert image_features.shape[-2] == 729
                image_features.append(image_feature)
                padding_masks.append(padding_mask)
                sizes.append(size)
                pos_trackings.append(pos_tracking)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = image_forward_outs.hidden_states[-1].to(images.dtype)
            padding_masks = image_forward_outs.padding_masks[-1]
            sizes = image_forward_outs.sizes[-1]
            pos_trackings = image_forward_outs.pos_trackings[-1]

        return image_features, padding_masks, sizes, pos_trackings

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        for p in self.vision_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.vision_tower.parameters():
            return p.device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size
        # return self.model_config["vision_cfg"]["image_size"] // self.model_config["vision_cfg"]["patch_size"]

    @property
    def image_size(self):
        return self.config.image_size
