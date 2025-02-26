import os
import torch
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image
import tqdm
import open_clip
import json

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
        image_tensor = self.preprocess(image)
        return image_tensor

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
    
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, precision='bf16')
    model.train()
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
