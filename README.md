#Modified code From [X-R1](https://github.com/dhcode-cpp/X-R1)

## Feature

- 🔥support QLoRA GRPO Trainning you can save gpu memory about 30%

## Result

Training the Qwen2-0.5B model with a stack of DeepSpeed ZeRO-3, QLoRA, GRPO, VLLM, and gradient checkpointing reduces GPU memory usage by 30% compared to a similar setup using standard LoRA. This memory saving is achieved with virtually no impact on training speed.

With QLoRA (4-bit, double quantization, nf4, gradient checkpointing)

<p align="left">
<img width="644" height="272" alt="Pasted image 20250715115423" src="https://github.com/user-attachments/assets/43f539d4-8639-45d2-928c-0b4d6fe4cd38" />
</p>

With QLoRA (4-bit, gradient checkpointing)

<p align="left">
<img width="647" height="280" alt="Pasted image 20250715102122" src="https://github.com/user-attachments/assets/53f4907f-4952-46e6-9669-feeb66ddd4d2" />
</p>

With gradient checkpointing (Enabling gradient checkpointing saves approximately 20% of GPU memory, but it comes at the cost of a 30% reduction in training speed.)

<p align="left">
<img width="640" height="278" alt="Pasted image 17525510581488" src="https://github.com/user-attachments/assets/97fd1501-d1f5-4da0-a930-d30dc9e0991b" />
</p>

baseline

<p align="left">
<img width="644" height="276" alt="Pasted image 20250714185922" src="https://github.com/user-attachments/assets/9cd0a5f4-8e03-416d-a351-364d805b0667" />
</p>
### training detail

<img width="1503" height="626" alt="image" src="https://github.com/user-attachments/assets/23015734-291b-4304-94ea-8775366bac82" />

### Benchmark Results (GSM8K)

| Model Version | GSM8K Avg. Score | gaokao2023en Avg. Score |
| :----------------------- | :----------------: | :----------------: |
| `Qwen2-0.5B` | **14.2** | **8.8** |
| `Qwen2-0.5B-X-R1` (Standard LoRA) | **24.7** | **14.5** |
| `Qwen2-0.5B-X-R1-QLoRA-V3` (QLoRA) | **26.2** | **16.9** |

Running Scripts:

### Example: GRPO + QLoRA

1. start vllm server:
```bash
PORT=8000
# Set the path to your LLM model
MODEL_PATH=Qwen2.5-0.5B
# Set the target GPU card index you want to clear and use (e.g., 4 for GPU 4)
GPU=5
VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=$GPU trl vllm-serve \
    --model $MODEL_PATH \
    --enforce-eager \
    --dtype bfloat16 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.9 \
    --max-model-len 8000 \
    --enable_lora \
    --max_lora_rank=512 \
    --max_cpu_loras=1 \
    --log_level "debug" \
    --port $PORT

echo "VLLM server command has been executed."
```

2. start trainer:
```bash
OUTPUT_MODEL="Qwen2.5-0.5B"
OUTPUT_MODEL2="Qwen2.5-0.5B-X-R1-QloraV4"
CHECKPOINT_DIR2="X-R1-test-QloraV4"

CUDA_VISIBLE_DEVICES="2,3,4" ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 7832 \
    --config_file recipes/zero3.yaml \
    --num_processes=3 src/x_r1/sgrpo.py \
    --config recipes/X_R1_zero_0dot5B_peft_config_4bit.yaml \
    --model_name_or_path=${OUTPUT_MODEL} \
    --output_dir=${CHECKPOINT_DIR2} \
    > output/x_r1_0dotB_sampling.log 2>&1
```

3. merge model
```bash
# Automatically find the latest checkpoint folder based on modification time
LATEST_CHECKPOINT=$(ls -td ${CHECKPOINT_DIR2}/checkpoint-* | head -n 1)

# If no checkpoint is found, exit
if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "No checkpoint found in ${CHECKPOINT_DIR}"
    exit 1
fi

# Print the latest checkpoint folder for logging
echo "Using latest checkpoint: $LATEST_CHECKPOINT"

CUDA_VISIBLE_DEVICES=4 python merge_with_lora_lowmem-info.py \
    --input_model ${OUTPUT_MODEL} \
    --input_lora ${LATEST_CHECKPOINT} \
    --output_model ${OUTPUT_MODEL2}
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
QQ: 2012456373
wechat a2012456373

## Acknowledge

[Open-R1](https://github.com/huggingface/open-r1), [TRL](https://github.com/huggingface/trl)

## Citation

```bib
@misc{Minami-su2025grpo-qlora,
  author = {Minami-su},
  title = {deepspeed-grpo-qlora-vllm},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Minami-su/deepspeed-grpo-qlora-vllm}}
  year = {2025},
}
```
