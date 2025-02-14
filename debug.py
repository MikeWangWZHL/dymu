# import torch
# from PIL import Image
# import open_clip

# model_name = "ViT-B-16-SigLIP-384"
# tome_model_name = "ViT-B-16-SigLIP-384-tome"

# model, _, preprocess = open_clip.create_model_and_transforms(tome_model_name, pretrained='webli')
# model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
# tokenizer = open_clip.get_tokenizer(model_name)

# image = preprocess(Image.open("docs/CLIP.png")).unsqueeze(0)
# text = tokenizer(["a diagram", "a dog", "a cat"])

# # with torch.no_grad(), torch.cuda.amp.autocast():
# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
#     image_features /= image_features.norm(dim=-1, keepdim=True)
#     text_features /= text_features.norm(dim=-1, keepdim=True)

#     text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
# # import pdb; pdb.set_trace()

# visual_encoder = model.visual.trunk
# visual_encoder.eval()
# feat_before_pooling, padding_mask = visual_encoder.forward_features(image)
# import pdb; pdb.set_trace()




###
import torch
import timm
import open_clip
from PIL import Image

model_name = "ViT-B-16-SigLIP-384-tome"
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='webli')
# model.eval()
image = preprocess(Image.open("docs/CLIP.png")).unsqueeze(0) # (1, 3, 384, 384)

# image = torch.randn(bs, 3, 384, 384)


bs = image.shape[0]
# ===== loading with pretrained weights using open_clip ====
tome_vision_encoder = model.visual.trunk
feat_before_pooling, padding_mask, size = tome_vision_encoder.forward_features(image)
print(feat_before_pooling.shape, size.shape)
if padding_mask is not None:
    print("num removed token in batch:", 576 * bs - (padding_mask==0).sum())
# 


tokenizer = open_clip.get_tokenizer(model_name)
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