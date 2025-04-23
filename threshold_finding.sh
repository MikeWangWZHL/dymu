
MODEL_NAME="ViT-L-14-336-tome-72out" # example name; see all avaliable model names at src/open_clip/model_configs/*-tome-*.json
PRETRAINED="openai" # open_clip pretrained name
JSON="<path to data.json>"
IM_BASE_PATH="<path to image_folder>"
SAVE_PATH="<path to checkpoints model.pt>"
BATCH_SIZE=64

# export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 -m open_clip_train.threshold_finder \
  --model $MODEL_NAME \
  --pretrained $PRETRAINED \
  --json_path $JSON \
  --im_base_path $IM_BASE_PATH \
  --batch_size $BATCH_SIZE \
  --save_path $SAVE_PATH