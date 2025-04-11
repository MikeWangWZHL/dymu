from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import copy
import torch

import warnings
warnings.filterwarnings("ignore")

from llava.model.multimodal_encoder.tome_encoder import SigLipVisionModelTome, SigLipVisionConfigTome, SigLipImageProcessor

def find_thresholds(
        model_name_or_path_pth,
        overwrite_config = None, 
        device_map = "auto",   
    ):
    config = SigLipVisionConfigTome()
    processor = SigLipImageProcessor()
    
    if overwrite_config is not None:
        for key in overwrite_config:
            if hasattr(config, key):
                setattr(config, key, overwrite_config[key])
    
    # Step 1: Initialize model with updated self.config
    model = SigLipVisionModelTome(config)  # Directly pass self.config

    # Step 2: Load pretrained weights from checkpoint
    checkpoint = torch.load(model_name_or_path_pth, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)  # Adjust based on checkpoint structure

    # post process some key mapping
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    # Step 3: Load the state_dict into the initialized model
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)  # `strict=False` allows partial mismatches
    # Print out unmatched keys
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    blocks = model.vision_model.encoder.layers
    learned_thresholds = []
    for i, block in enumerate(blocks):
        # print(block.threshold)
        learned_thresholds.append(block.threshold.item())
    return learned_thresholds
    


## path to the saved encoder checkpoint for getting the thresholds ##
threshold_checkpoint_path = "checkpoints/threshold_checkpoints/siglip-so400m-patch14-384-tome-81out.pth"

# path to the pretrained checkpoint with config.json modified
pretrained = "vlm_checkpoints/llava-onevision-qwen2-7b-si"


model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
llava_model_args = {
    "multimodal": True,
    "attn_implementation": "sdpa",
}

learned_thresholds = find_thresholds(threshold_checkpoint_path)
overwrite_config = {
    "r_total": 405,
    "r_schedule": "constant",
    # "merge_mode": "instance_level",
    "merge_mode": "batch_level",
    "repeat_merged_tokens": True, # whether enabling Virtual Token Unmerging (VTU)
    "specified_thresholds": learned_thresholds
}
print("learned thresholds:", learned_thresholds)

tokenizer, model, image_processor, max_length = load_pretrained_model(
    pretrained, None, model_name, tome_kwargs=None,
    device_map=device_map, overwrite_config=overwrite_config, **llava_model_args)  # Add any other thing you want to pass in llava_model_args

model.eval()

image = Image.open("LLaVA/images/001.jpg")
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
model = model.to(device, dtype=torch.float16)

conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
question = DEFAULT_IMAGE_TOKEN + "\nDescribe the image."
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size]

cont = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_new_tokens=4096,
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
print(text_outputs[0])
import pdb; pdb.set_trace()