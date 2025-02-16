### debugging 
###
import torch
import timm
import open_clip
from PIL import Image

model_name = "ViT-B-16-SigLIP-384-tome"
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='webli')
# model.eval()
# img_paths = [
#     "/shared/nas/data/m1/wangz3/salesforce_intern/llava_1_5_data/stage1/00453/004539375.jpg",
#     "/shared/nas/data/m1/wangz3/salesforce_intern/llava_1_5_data/stage1/00594/005947502.jpg"
# ]
img_paths = [
    "/shared/nas/data/m1/wangz3/salesforce_intern/llava_1_5_data/stage1/00453/004539375.jpg",
    "/shared/nas/data/m1/wangz3/salesforce_intern/llava_1_5_data/stage1/00511/005116462.jpg"
]
# img_paths = [
#     "/shared/nas/data/m1/wangz3/salesforce_intern/llava_1_5_data/stage1/00453/004539375.jpg",
#     "/shared/nas/data/m1/wangz3/salesforce_intern/llava_1_5_data/stage1/00223/002239345.jpg"
# ]

# image = preprocess(Image.open("docs/CLIP.png")).unsqueeze(0)
image = [preprocess(Image.open(p)) for p in img_paths] # (2, 3, 384, 384)
image = torch.stack(image, dim=0)

tome_vision_encoder = model.visual.trunk
tome_vision_encoder.to(dtype=torch.bfloat16)
image = image.to(dtype=torch.bfloat16)

for name, param in tome_vision_encoder.named_parameters():
    print(f"{name}: {param.dtype}")

outputs = tome_vision_encoder.forward_features_all_layers(image)

hidden_states = outputs.hidden_states
padding_masks = outputs.padding_masks # b, n
print((padding_masks[-1]==0).sum())
import pdb; pdb.set_trace()
# image = torch.randn(bs, 3, 384, 384)



# ###
# import torch
# import timm
# import open_clip
# from PIL import Image

# model_name = "ViT-B-16-SigLIP-384-tome"
# model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='webli')
# # model.eval()
# image = preprocess(Image.open("docs/CLIP.png")).unsqueeze(0) # (1, 3, 384, 384)

# # image = torch.randn(bs, 3, 384, 384)


# bs = image.shape[0]
# # ===== loading with pretrained weights using open_clip ====
# tome_vision_encoder = model.visual.trunk
# feat_before_pooling, padding_mask, size = tome_vision_encoder.forward_features(image)
# print(feat_before_pooling.shape, size.shape)
# if padding_mask is not None:
#     print("num removed token in batch:", 576 * bs - (padding_mask==0).sum())
# # 


# tokenizer = open_clip.get_tokenizer(model_name)
# text = tokenizer(["a diagram", "a dog", "a cat"])

# # with torch.no_grad(), torch.cuda.amp.autocast():
# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
#     image_features /= image_features.norm(dim=-1, keepdim=True)
#     text_features /= text_features.norm(dim=-1, keepdim=True)

#     text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
# import pdb; pdb.set_trace()