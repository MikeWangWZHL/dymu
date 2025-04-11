<!-- <p align="center">
  <img src="static/images/logo.png" alt="logo" width="500">
</p> -->

<div align="center">
  <h1>DyMU: Dynamic Merging and Virtual Unmerging for Efficient VLMs</h1>
</div>

<p align="center">
<a href="">🌐 Homepage</a>
•
<a href="">🗃️ arXiv</a>
•
<a href="">📃 PDF </a>
•
<a href="">💻 Code</a>
•
<a href="https://huggingface.co/mikewang/DyMU/tree/main" >🤗 Models</a>


<div align="center">
Zhenhailong Wang<sup>1*</sup>, Senthil Purushwalkam<sup>2*</sup>, Caiming Xiong<sup>2</sup>, 
Silvio Savarese<sup>2</sup>, Heng Ji<sup>1</sup>, Ran Xu<sup>2</sup>
</div>
<br>
<div align="center">
<sup>1</sup>University of Illinois Urbana-Champaign   <sup>2</sup>Salesforce Research
</div>
<div align="center">
<sup>*</sup>Equal Contribution
</div>
<br>

<p align="center">
  <img src="static/images/teaser_long.png" alt="overview" width="900">
</p>


## Installation

### Minimal setup
This allows using DyMU encoders to obtain dynamic length visual features.
```
pip install -e .
```

### VLM specific setup

1. install the llava/llava-one-vision package following:
    - if using llava-1.5
      ```
      conda create -n llava python=3.10 -y
      conda activate llava
      cd LLaVA
      pip install --upgrade pip
      pip install -e ".[train]"
      ```
    - if using llava-one-vision
      ```
      conda create -n llava_next python=3.10 -y
      conda activate llava_next
      cd LLaVA-NeXT
      pip install --upgrade pip
      pip install -e ".[train]"
      ```
2. Upgrade several pip modules for compatibility with open_clip:
    ```
    pip install --upgrade transformers accelerate sentencepiece deepspeed peft
    pip install line-profiler
    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
    pip install --upgrade timm ipdb
    ```

3. Install the custom open_clip
    ```
    cd .. # cd to the root of the repo
    pip install -e .
    ```

  
## Threshold Finding with DToMe
TODO: senthil please add details

## Inference

Download DyMU encoder checkpoints with pre-computed from [here](https://huggingface.co/mikewang/DyMU/tree/main).
Or run threshold finding as described in [here](#threshold-finding-with-dtome).
Put the encoder checkpoints under `checkpoints/threshold_checkpoints`

### Dynamic length visual encoding usage examples
- DyMU with Siglip encoder example:
  ```
  python inference_siglip.py
  ```

- DyMU with OpenAI CLIP encoder example:
  ```
  python inference_openai_clip.py
  ```

### VLM inference with DyMU encoders

Make sure the VLM specific installation for the expected VLM is done as described [here](#vlm-specific-setup).

#### Llava-1.5:
  - Download pretrained llava-1.5 checkpoint, e.g., https://huggingface.co/liuhaotian/llava-v1.5-7b, and put it under `checkpoints/vlm_checkpoints`.
  - Modify the `mm_vision_tower` field in the `config.json` to  `ViT-L-14-336-tome-72out` for pointing the model to use DyMU vision tower. (72out here is only a template, one can use any thresholds during inference)
  - Run inference example:
    ```
    conda activate llava
    CUDA_VISIBLE_DEVICES=0 python LLaVA/inference_dymu_llava.py
    ```

#### Llava-One-Vision:
  - Download pretrained llava-ov checkpoint, e.g., https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-si, and put it under `checkpoints/vlm_checkpoints`.
  - Modify the `mm_vision_tower` field in the `config.json` to  `ViT-SO400M-14-SigLIP-384-tome` for pointing the model to use DyMU vision tower.
  - Run inference example:
    ```
    conda activate llava_next
    CUDA_VISIBLE_DEVICES=0 python LLaVA-NeXT/inference_dymu_llava_ov.py
    ```

### Implementation Notes
- In the paper, we analyze the efficiency gain with Virtual Token Unmerging (VTU) on both MLP layers and Self-Attention layers. Although it theoretical holds that VTU can bring additional speed up during self-attention, we find that in practice, the customized attention operation is slower than directly using matmul on the expanded sequence (where merged tokens are repeated). This is because the .matmul function in PyTorch is highly optimized for long sequences — the overhead caused by the increased number of individual multiplications in the decomposed attention outweighs the benefits of reduced matrix dimensions. Therefore, by default, we use the faster, simple implementation. Nevertheless, for reference, we also include an implementation that strictly follows the VTU decomposition described in the paper, located in `LLaVA/llava/model/language_model/vtu_attention_exact_imple.py`, where we provide the `CustomLlamaAttention` class. We encourage readers to explore further improvements to its efficiency.
- For LLaVA-One-Vision, the input to the encoder is a batch of image crops. In DyMU, since each crop can have a variable number of remaining tokens after each layer, we need to pad the sequences, which introduces additional computational overhead. This can make DyMU on LLaVA-One-Vision noticeably slower than ToMe, which performs instance-level merging and maintains a constant number of tokens at each layer.
