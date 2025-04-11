### debugging 
###
import torch
import timm
import open_clip
from PIL import Image
import os

model_name = "ViT-B-16-SigLIP-384-tome-72out" # see avaliable model names at src/open_clip/model_configs/*-SigLIP-*-tome*.json
threshold_checkpoint_path = "checkpoints/threshold_checkpoints/ViT-B-16-SigLIP-384-tome-72out.pth"

## loading a saved checkpoint with found thresholds ##
timm_kwargs = {
    'pretrained': threshold_checkpoint_path,
    'merge_mode': "batch_level"
}
####

## loading an original checkpoint without threshold finding ##
# timm_kwargs = {
#     'pretrained': "webli",
#     # 'merge_mode': "instance_level", # => set this for ToMe merging, i.e., merging constant num of tokens per layer
#     'merge_mode': "batch_level", # => set this for DyMU merging
#     "r_total":504, # => avg total number of tokens to merge per instance; note that: if model is set to eval mode, merging will be based on threshold not the r_total
#     # "specified_thresholds":[0.98046875, 0.90234375, 0.875, 0.8828125, 0.875, 0.8828125, 0.890625, 0.890625, 0.8828125, 0.890625, 0.875, 0.875] # => one can also directly specify the thresholds
# }
####

model, _, preprocess = open_clip.create_model_and_transforms(model_name, **timm_kwargs)
model.eval() # if set to eval mode, merging will be based on threshold; otherwise, will be based on r_total

current_thresholds = []
for i, block in enumerate(model.visual.trunk.blocks):
    current_thresholds.append(block.threshold.item())
print("thresholds:", current_thresholds)



## visual encoding usage example ##
image =  preprocess(Image.open("docs/CLIP.png")).unsqueeze(0)
tome_vision_encoder = model.visual.trunk
outputs = tome_vision_encoder.forward_features_all_layers(image)

hidden_states = outputs.hidden_states # tuple with a length of num_layers
print("Last layer hidden_states shape (1, num_remaining_tokens, D):", hidden_states[-1].shape)
print()


## image-text matching example ##
image = preprocess(Image.open("docs/CLIP.png")).unsqueeze(0) # (1, 3, 384, 384)
tokenizer = open_clip.get_tokenizer(model_name)
text_candidates = ["a diagram", "a dog", "a cat"]
text = tokenizer(text_candidates)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:")
for i in range(len(text_probs[0])):
    print(f"{text_candidates[i]}: {text_probs[0][i]}")