#Modified code From [X-R1](https://github.com/dhcode-cpp/X-R1)

## Feature

- 🔥support QLoRA GRPO Trainning you can save gpu memory about 30%

## Result
训练模型Qwen2.5-0.5b，使用deepspeed-zero3+Qlora+GRPO+vllm+gradient_checkpointing的gpu内存使用量，相比deepspeed-zero3+lora+GRPO+vllm+gradient_checkpointing，的显存使用量能够节省30%，速度基本保持不变

使用了load in 4bit  double quant nf4

<p align="left">
<img width="644" height="272" alt="Pasted image 20250715115423" src="https://github.com/user-attachments/assets/43f539d4-8639-45d2-928c-0b4d6fe4cd38" />
</p>

使用了load in 4bit 

<p align="left">
<img width="647" height="280" alt="Pasted image 20250715102122" src="https://github.com/user-attachments/assets/53f4907f-4952-46e6-9669-feeb66ddd4d2" />
</p>

未使用Qlora的

<p align="left">
<img width="640" height="278" alt="企业微信截图_17525510581488" src="https://github.com/user-attachments/assets/97fd1501-d1f5-4da0-a930-d30dc9e0991b" />
</p>

未使用Qlora，未使用gradient_checkpointing

<p align="left">
<img width="644" height="276" alt="Pasted image 20250714185922" src="https://github.com/user-attachments/assets/9cd0a5f4-8e03-416d-a351-364d805b0667" />
</p>

### Overview
Qwen2.5-0.5B-X-R1 gsm8k   avg  
			21.2    21.2
Qwen2.5-0.5B-X-R1-QloraV3 gsm8k   avg  
			26.2    26.2 
Running Scripts:

```bash
bash ./scripts/run_x_r1_zero.sh
```


### Example: GRPO + QLoRA

1. multi-gpu run:

```bash
CUDA_VISIBLE_DEVICES="2,3,4" ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 7832 \
    --config_file recipes/zero3.yaml \
    --num_processes=3 src/x_r1/sgrpo.py \
    --config recipes/X_R1_zero_0dot5B_peft_config_4bit.yaml \
    --model_name_or_path=${OUTPUT_MODEL} \
    --output_dir=${CHECKPOINT_DIR2} \
    > output/x_r1_0dotB_sampling.log 2>&1
```

## Installation

### conda & pip

required: cuda >= 12.4

```bash
conda create -n test python=3.11
conda activate test
```

and

"Note: You must install the specific version of trl from requirements.txt, as it has been modified."

```bash
pip install -r requirements.txt
pip install flash-attn
```

## Todo
- semi online GRPO Trainer
- GRPO with MCTSR-Zero
- 
## About

If you have any suggestions, please contact: 2012456373@qq.com

## Acknowledge

[Open-R1](https://github.com/huggingface/open-r1), [TRL](https://github.com/huggingface/trl)

## Citation

```bib
@misc{deng2025xr1,
  author = {hang deng},
  title = {X-R1},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/dhcode-cpp/X-R1}}
  year = {2025},
}
```
