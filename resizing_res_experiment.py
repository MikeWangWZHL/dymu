### debugging 
###
import torch
import timm
import open_clip
from PIL import Image
import os

def load_tome_clip_model():
    # model_name = "ViT-L-14-336-tome-72out" # with quickgelu
    # model_name = "ViT-L-14-336-tome-192out" # with quickgelu
    model_name = "ViT-L-14-336-tome-384out" # with quickgelu
    # model_name = "ViT-B-16-SigLIP-384-tome-72out"
    # model_name = "ViT-B-16-SigLIP-384-tome-192out"
    # model_name = "ViT-B-16-SigLIP-384-tome-384out"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, 
        pretrained=f"/shared/nas2/wangz3/salesforce_intern_nas2/open_clip_merging/LLaVA/checkpoints/shared_by_senthil/tome_nofinetune_clsbugfix/threshold_checkpoints/{model_name}.pth"
    )
    model.eval()
    tome_vision_encoder = model.visual.trunk if hasattr(model.visual, "trunk") else model.visual
    if "siglip" in model_name.lower():
        crop_size = 384
    else:
        crop_size = 336
    return tome_vision_encoder, preprocess, crop_size


def count_avg_remaining_tokens_w_crop(img_paths, tome_vision_encoder, preprocess, device, dtype, crop_size=None,additional_resize=None):
    images = []
    for p in img_paths:
        img = Image.open(p).convert("RGB")  # Ensure it's in RGB mode
        
        # Apply center crop if crop_size is given and the image is larger
        if crop_size is not None:
            w, h = img.size
            if w > crop_size and h > crop_size:
                left = (w - crop_size) // 2
                top = (h - crop_size) // 2
                img = img.crop((left, top, left + crop_size, top + crop_size))
        
        # Apply additional resizing if specified
        if additional_resize is not None:
            img = img.resize((additional_resize, additional_resize), Image.BICUBIC)
        
        # Preprocess the image
        images.append(preprocess(img))
    
    input_pixels = torch.stack(images, dim=0).to(device, dtype=dtype)
    bsz = input_pixels.size(0)
    with torch.no_grad():
        outputs = tome_vision_encoder.forward_features_all_layers(input_pixels)
    
    padding_masks = outputs.padding_masks
    last_padding_mask = padding_masks[-1]
    total_remaining_tokens = (last_padding_mask < 0.5).sum()
    avg_remaining_tokens = total_remaining_tokens / bsz
    return avg_remaining_tokens, outputs

from glob import glob
if __name__ == "__main__":
    # load llava-bench images
    llava_bench_image_dir = "/shared/nas2/wangz3/salesforce_intern_nas2/llava_1_5_data/eval/llava-bench-in-the-wild/images"
    img_paths = glob(os.path.join(llava_bench_image_dir, "*.jpg"))

    # load model and preprocess
    tome_vision_encoder, preprocess, crop_size = load_tome_clip_model()
    device = "cuda:2"
    dtype = torch.float32
    tome_vision_encoder.to(device)
    tome_vision_encoder.to(dtype=dtype)
    
    print(f"original crop_size: {crop_size}")
    
    # count avg remaining tokens in original size
    avg_remaining_tokens, outputs = count_avg_remaining_tokens_w_crop(
        img_paths, tome_vision_encoder, preprocess, device, dtype, crop_size=crop_size
    )
    print(f"avg remaining tokens in original size: {avg_remaining_tokens}")
    
    # downsize first
    additional_resize = 112
    avg_remaining_tokens, outputs = count_avg_remaining_tokens_w_crop(
        img_paths, tome_vision_encoder, preprocess, device, dtype, crop_size=crop_size, additional_resize=additional_resize)
    print(f"avg remaining tokens with additional resize: {additional_resize}: {avg_remaining_tokens}")
    
    
    # downsize first
    additional_resize = 56
    avg_remaining_tokens, outputs = count_avg_remaining_tokens_w_crop(
        img_paths, tome_vision_encoder, preprocess, device, dtype, crop_size=crop_size, additional_resize=additional_resize)
    print(f"avg remaining tokens with additional resize: {additional_resize}: {avg_remaining_tokens}")