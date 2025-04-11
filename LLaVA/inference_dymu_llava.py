from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import open_clip
import os

def find_thresholds(threshold_finding_checkpoint):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14-336-tome-72out", # can be any "ViT-L-14-336-tome-*" 
        pretrained=threshold_finding_checkpoint
    )
    tome_vision_encoder = model.visual.trunk if hasattr(model.visual, "trunk") else model.visual
    blocks = tome_vision_encoder.blocks if hasattr(tome_vision_encoder, "blocks") else tome_vision_encoder.transformer.resblocks
    learned_thresholds = []
    for i, block in enumerate(blocks):
        # print(block.threshold)
        learned_thresholds.append(block.threshold.item())
    return learned_thresholds


## path to the saved encoder checkpoint for getting the thresholds ##
threshold_checkpoint_path = "checkpoints/threshold_checkpoints/ViT-L-14-336-tome-72out.pth"
# threshold_checkpoint_path = "/shared/nas2/wangz3/salesforce_intern_nas2/open_clip_merging/LLaVA/checkpoints/shared_by_senthil/tome_nofinetune_clsbugfix/threshold_checkpoints/ViT-L-14-336-tome-72out.pth"

# path to the pretrained checkpoint with config.json modified
model_path = "vlm_checkpoints/llava-v1.5-7b"

model_name = "llava_v1.5_tome_openai"

## DyMU config with VTU ##
learned_thresholds = find_thresholds(threshold_checkpoint_path)
tome_kwargs = {
    'pretrained': "openai",
    'pretrained_origin_tag': "openai",
    'merge_mode': "batch_level",
    "repeat_merged_tokens": True, # whether enabling Virtual Token Unmerging (VTU)
    "r_total":504,
    "specified_thresholds": learned_thresholds
}

print("learned thresholds:", learned_thresholds)

## ToMe config with VTU ##
# tome_kwargs = {
#     'pretrained': "openai",
#     'pretrained_origin_tag': "openai",
#     'merge_mode': "instance_level",
#     "r_total":504, # 72 out at final layer
#     "repeat_merged_tokens": True # whether enabling Virtual Token Unmerging (VTU)
# }

prompt = "Describe the image."
image_file = "LLaVA/images/001.jpg"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": model_name,
    "query": prompt,
    "conv_mode": "llava_v1",
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512,
    "tome_kwargs": tome_kwargs
})()

eval_model(args)