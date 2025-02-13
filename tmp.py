###
# Try a siglip model from scratch

import timm
from src.open_clip.tome import *

target_avg_total_remove_tokens = 96
model = timm.create_model('vit_base_patch16_siglip_224_tome', pretrained=False, r_total=target_avg_total_remove_tokens)

import torch
bs = 2
dummy_input = torch.randn(bs, 3, 224, 224)
proj_feat = model(dummy_input)
print(proj_feat.shape)
feat_before_pooling, padding_mask = model.forward_features(dummy_input)
print(feat_before_pooling.shape, padding_mask.shape)
if padding_mask is not None:
    print("num removed token in batch:", 196 * bs - (padding_mask==0).sum(), "| expected: ", target_avg_total_remove_tokens * bs)

print("\n=====================================================\n")
###
# Try loading a pretrained clip l-14 336 with our ToME class (but seems not exactly the openai's checkpoint, need double check when using it with OpenCLIP)
model = timm.create_model('vit_large_patch14_clip_336_tome', pretrained=True, r_total=target_avg_total_remove_tokens) # class token == True for pretrained
print("Loaded pretrained model: vit_large_patch14_clip_336 | (the CLIP version used in llava-1.5)")
# print(model)
dummy_input = torch.randn(2, 3, 336, 336)
feat_before_pooling, padding_mask = model.forward_features(dummy_input)
print(feat_before_pooling.shape, padding_mask.shape)
if padding_mask is not None:
    print("num removed token in batch:", 577 * bs - (padding_mask==0).sum(), "| expected: ", target_avg_total_remove_tokens * bs)