import json
import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None, **kwargs):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


import open_clip
from open_clip.tome import ToMEVisionTransformer, CLIPVisionEncoderToMEOutput
from open_clip.transformer import ToMEOpenAIVisionTransformer
from open_clip.factory import get_model_config
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_processing_base import BatchFeature
from typing import List, Dict, Optional, Union
from PIL import Image
import numpy as np
from open_clip.transform import PreprocessCfg

class ToMeImageProcessor(BaseImageProcessor):
    """
    Constructs an OpenCLIP image processor using the `preprocess` transform from OpenCLIP.
    """
    model_input_names = ["pixel_values"]

    def __init__(self, 
        preprocess_transform, 
        size: Optional[Dict[str, int]] = None,
        image_mean: Optional[List[float]] = [0.5, 0.5, 0.5],
        image_std: Optional[List[float]] = [0.5, 0.5, 0.5],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.transform_fn = preprocess_transform

        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size
        self.crop_size = size
        
    def preprocess(
        self,
        images: Union[Image.Image, List[Image.Image]],
        return_tensors: Optional[str] = "pt",
        **kwargs,
    ) -> BatchFeature:
        if not isinstance(images, list):
            images = [images]
        processed_images = [self.transform_fn(image) for image in images]
        
        if return_tensors == "pt":
            pixel_values = torch.stack(processed_images)
        else:
            pixel_values = np.stack([img.numpy() for img in processed_images])

        return BatchFeature(data={"pixel_values": pixel_values})

class CLIPVisionTowerToMe(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        print(f"CLIPVisionTowerToMe select_layer: {self.select_layer}, select_feature: {self.select_feature}")
        
        self.vision_tower_kwargs = getattr(args, 'tome_vision_tower_kwargs', {})
        if isinstance(self.vision_tower_kwargs, str):
            self.vision_tower_kwargs = json.loads(self.vision_tower_kwargs)

        if not delay_load:
            self.load_model(**self.vision_tower_kwargs)
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model(**self.vision_tower_kwargs)
        else:
            self.cfg_only = {"vision_tower_config": get_model_config(self.vision_tower_name), "tome_vision_tower_kwargs": self.vision_tower_kwargs}

    def load_model(self, device_map=None, device=None, dtype=None, **kwargs):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        self.vision_tower_kwargs.update(kwargs)
        
        print(f'Loading ToME vision transformer model name: {self.vision_tower_name}')
        print(f'ToME vision transformer updated args: {self.vision_tower_kwargs}')

        if "repeat_merged_tokens" in self.vision_tower_kwargs:
            self.repeat_merged_tokens = self.vision_tower_kwargs["repeat_merged_tokens"]
        else:
            self.repeat_merged_tokens = False
        print(f"Setting 'repeat_merged_tokens' to: {self.repeat_merged_tokens}")
        
        model, _, _ = open_clip.create_model_and_transforms(self.vision_tower_name, **self.vision_tower_kwargs)
        tome_vision_encoder = model.visual.trunk if hasattr(model.visual, 'trunk') else model.visual
        assert isinstance(tome_vision_encoder, ToMEVisionTransformer) or isinstance(tome_vision_encoder, ToMEOpenAIVisionTransformer)
        self.vision_tower = tome_vision_encoder
        self.vision_tower.eval()
        self.vision_tower.requires_grad_(False)

        if device is not None:
            self.vision_tower.to(device=device)
        if dtype is not None:
            self.vision_tower.to(dtype=dtype)

        ## ensure preprocess is the same as the original pretrained model ##
        assert "pretrained_origin_tag" in self.vision_tower_kwargs, f"pretrained_origin_tag not found in {self.vision_tower_kwargs}"
        pretrained_origin_model, _, preprocess = open_clip.create_model_and_transforms(self.vision_tower_name, 
                                                                 pretrained=self.vision_tower_kwargs['pretrained_origin_tag'])
        pp_cfg = PreprocessCfg(**pretrained_origin_model.visual.preprocess_cfg)
        print("tome preprocess cfg:", pp_cfg)
        self.image_processor = ToMeImageProcessor(
            preprocess_transform=preprocess,
            size={"height":pp_cfg.size[0], "width":pp_cfg.size[1]},
            image_mean=pp_cfg.mean,
            image_std=pp_cfg.std,
        )
        self.is_loaded = True

    def feature_select(self, image_forward_outs: CLIPVisionEncoderToMEOutput):
        image_feature = image_forward_outs.hidden_states[self.select_layer] # b, n, d
        padding_mask = image_forward_outs.padding_masks[self.select_layer] # b, n
        size = image_forward_outs.sizes[self.select_layer] # b, n, 1
        pos_tracking = image_forward_outs.pos_trackings[self.select_layer] # b, n, 2
        if self.vision_tower.has_class_token:
            if self.select_feature == 'patch':
                image_feature = image_feature[:, 1:]
                if size is not None:
                    size = size[:, 1:]
                if padding_mask is not None:
                    padding_mask = padding_mask[:, 1:]
                if pos_tracking is not None:
                    pos_tracking = pos_tracking[:, 1:, 1:] # remove class token; row and col; so that the row 1s sum up to (original size - 1)
            # elif self.select_feature == 'cls_patch':
            #     pass
            else:
                raise ValueError(f'Unexpected select feature: {self.select_feature} for this Tome encoder:', self.vision_tower_name)
        else:
            pass # use all patches
        return image_feature, padding_mask, size, pos_tracking

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            padding_masks = []
            sizes = []
            pos_trackings = []
            for image in images:
                image_forward_out = self.vision_tower.forward_features_all_layers(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                image_feature, padding_mask, size, pos_tracking = self.feature_select(image_forward_out)
                # image_features.append(image_feature.to(image.dtype))
                image_features.append(image_feature)
                padding_masks.append(padding_mask)
                sizes.append(size)
                pos_trackings.append(pos_tracking)
        else:
            image_forward_outs = self.vision_tower.forward_features_all_layers(images.to(device=self.device, dtype=self.dtype))
            image_features, padding_masks, sizes, pos_trackings = self.feature_select(image_forward_outs)
            # image_features = image_features.to(images.dtype)
        return image_features, padding_masks, sizes, pos_trackings

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def config(self):
        if self.is_loaded:
            return get_model_config(self.vision_tower_name)
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.vision_tower.embed_dim

    @property
    def num_patches_per_side(self):
        if isinstance(self.vision_tower, ToMEOpenAIVisionTransformer):
            return self.vision_tower.img_size // self.vision_tower.patch_size
        else:
            return self.vision_tower.patch_embed.img_size // self.vision_tower.patch_embed.patch_size

    @property
    def num_patches(self):
        if isinstance(self.vision_tower, ToMEOpenAIVisionTransformer):
            return self.vision_tower.num_patches
        else:
            return self.vision_tower.patch_embed.num_patches
    
    def train(self, mode=True):
        super(CLIPVisionTowerToMe, self).train(mode=False)
        return 


class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None, **kwargs):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
