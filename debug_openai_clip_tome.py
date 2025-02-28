### debugging 
###
import torch
import timm
import open_clip
from PIL import Image
import os

### original open ai ###
# model_name = "ViT-L-14-336-quickgelu"
# model_name = "ViT-L-14-336-tome-72out" # with quickgelu
# tome_kwargs = {"specified_thresholds": [0.8]*24}
# model_name = "ViT-L-14-336-tome-72out" # with quickgelu
# tome_kwargs = {"r_total": 0}
tome_kwargs = {"r_total": 504}

# model_name = "ViT-L-14-336-tome-72out" # with quickgelu
# model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='/shared/nas2/wangz3/salesforce_intern_nas2/open_clip_merging/LLaVA/checkpoints/shared_by_senthil/tome_nofinetune/threshold_checkpoints/ViT-L-14-336-tome-72out.pth', **tome_kwargs)

model_name = "ViT-L-14-336-tome-72out-linear" # with quickgelu
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='/shared/nas2/wangz3/salesforce_intern_nas2/open_clip_merging/LLaVA/checkpoints/shared_by_senthil/tome_nofinetune/threshold_checkpoints/ViT-L-14-336-tome-72out-linear.pth', **tome_kwargs)

# model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='openai', **tome_kwargs)

# print()
model.eval()
# model.train()


### test clip ###
tokenizer = open_clip.get_tokenizer(model_name)
image = preprocess(Image.open("docs/CLIP.png")).unsqueeze(0)
text = tokenizer(["a diagram", "a dog", "a cat"])

# with torch.no_grad(), torch.cuda.amp.autocast():
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
import pdb; pdb.set_trace()

### check thresholds ###
tome_vision_encoder = model.visual.trunk if hasattr(model.visual, "trunk") else model.visual
blocks = tome_vision_encoder.blocks if hasattr(tome_vision_encoder, "blocks") else tome_vision_encoder.transformer.resblocks
learned_thresholds = []
for i, block in enumerate(blocks):
    # print(block.threshold)
    learned_thresholds.append(block.threshold.item())
print(learned_thresholds)
import pdb; pdb.set_trace()

img_paths = [
    "/shared/nas2/wangz3/salesforce_intern_nas2/llava_1_5_data/stage1/LLaVA-Pretrain/00453/004539375.jpg",
    "/shared/nas2/wangz3/salesforce_intern_nas2/llava_1_5_data/stage1/LLaVA-Pretrain/00511/005116462.jpg"
]
image = [preprocess(Image.open(p)) for p in img_paths] # (2, 3, 336, 336)
image = torch.stack(image, dim=0)

tome_vision_encoder = model.visual.trunk if hasattr(model.visual, "trunk") else model.visual
tome_vision_encoder.to("cuda:1")
tome_vision_encoder.to(dtype=torch.float16)
image = image.to("cuda:1")
image = image.to(dtype=torch.float16)

# for name, param in tome_vision_encoder.named_parameters():
#     print(f"{name}: {param.dtype}")

outputs = tome_vision_encoder.forward_features_all_layers(image)

hidden_states = outputs.hidden_states
padding_masks = outputs.padding_masks # b, n
if padding_masks[-1] is not None:
    print("remaining tokens: ", (padding_masks[-1]==0).sum())

import pdb; pdb.set_trace()

