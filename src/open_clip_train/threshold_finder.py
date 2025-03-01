import os
import ipdb
import torch
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image
import tqdm
import open_clip
import json
from llava.model.multimodal_encoder.tome_encoder import SigLipVisionModelTome, SigLipVisionConfigTome, SigLipImageProcessor

def setup_distributed():
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())

def cleanup_distributed():
    dist.destroy_process_group()

class ImageConversationDataset(Dataset):
    def __init__(self, base_path, json_path, preprocess):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.data = [d for d in self.data if "image" in d]
        
        self.base_path = base_path
        self.preprocess = preprocess

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            image_path = os.path.join(self.base_path, item["image"])
        except:
            print(item)
        image = Image.open(image_path).convert("RGB")
        try:
            image_tensor = self.preprocess(image)
        except:
            image_tensor = self.preprocess.preprocess(images=image, return_tensors="pt").pixel_values[0]
        return image_tensor



def load_siglip_tome_llava_ov(
        model_name_or_path = "google/siglip-so400m-patch14-384",
        overwrite_config = None, 
        device_map = "auto",   
    ):
    config = SigLipVisionConfigTome()
    processor = SigLipImageProcessor()
    
    if overwrite_config is not None:
        for key in overwrite_config:
            if hasattr(config, key):
                setattr(config, key, overwrite_config[key])

    print(f"Loading vision tower: {model_name_or_path} with SigLipVisionModelTome class.")
    print(f"Config: {config}")
    
    # Step 1: Initialize model with updated self.config
    model = SigLipVisionModelTome(config)  # Directly pass self.config

    # Step 2: Load pretrained weights from checkpoint
    state_dict = SigLipVisionModelTome.from_pretrained(model_name_or_path, device_map=device_map).state_dict()
    
    # Step 3: Load the state_dict into the initialized model
    model.load_state_dict(state_dict, strict=False)  # `strict=False` allows partial mismatches: thresholds

    if config.set_training_mode:
        model.train()
    else:
        model.eval()
    return model, processor



def parse_args():
    parser = argparse.ArgumentParser(description="Train an image conversation model with CLIP")
    parser.add_argument("--model", type=str, required=True, help="Model architecture to use")
    parser.add_argument("--pretrained", type=str, required=True, help="Pretrained model weights")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the JSON file containing image data")
    parser.add_argument("--im_base_path", type=str, required=True, help="Base directory where images are stored")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the trained model checkpoint")
    return parser.parse_args()

def main():
    args = parse_args()
    setup_distributed()
    rank = dist.get_rank()
    
    if args.model.startswith("google/siglip-so400m-patch14-384"):
            model_name_or_path = "google/siglip-so400m-patch14-384"
            rem_txt = args.model[len("google/siglip-so400m-patch14-384")+1:]
            assert rem_txt.split('-')[0][-3:] == "out"
            r = int(rem_txt.split('-')[0][:-3])
            if len(rem_txt.split('-'))>1:
                schedule = rem_txt.split('-')[1]
            else:
                schedule = "constant"
            tome_kwargs = {
                "r_total": r,
                "r_schedule": schedule,
                "set_training_mode": True,
            }
            print(f'Using tome kwargs as {tome_kwargs}')
            model, preprocess = load_siglip_tome_llava_ov(model_name_or_path=model_name_or_path, overwrite_config=tome_kwargs,device_map="cpu")
            # preprocess = lambda x: image_processor.preprocess(images=x, return_tensors="pt").pixel_values[0]
            model.train()
            model.to(torch.bfloat16)
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            args.model, pretrained=args.pretrained, precision='bf16')
        model.train()
    _, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-16-SigLIP-384", pretrained="webli", precision='bf16')
    model.cuda()
    model = DDP(model, device_ids=[rank], output_device=rank)
    
    dataset = ImageConversationDataset(args.im_base_path, args.json_path, preprocess)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
    
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="dataloader", disable=(rank != 0)):
            out = model(batch.to("cuda").to(torch.bfloat16))
    
    if rank == 0:
        checkpoint_dict = {"name": args.model, "state_dict": model.state_dict()}
        torch.save(checkpoint_dict, args.save_path)
        print(f'Saving model to {args.save_path}')
    
    cleanup_distributed()

if __name__ == "__main__":
    main()
