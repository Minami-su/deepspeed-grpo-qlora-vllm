#Modified code From [X-R1](https://github.com/dhcode-cpp/X-R1)

## Feature

- 🔥support QLoRA GRPO Trainning

## Result

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
conda create -n xr1 python=3.11
conda activate xr1
```

and

```bash
pip install -r requirements.txt
pip install flash-attn
```

### quick start

\[option\]: single GPU with LoRA:

```shell
ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero1.yaml \
--num_processes=1 \
src/x_r1/grpo.py \
--config recipes/X_R1_zero_0dot5B_peft_config.yaml \
> ./output/x_r1_test_sampling.log 2>&1
```

\[option\]Multi-GPU:

```shell
ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/accelerate_configs/zero3.yaml \
--num_processes=1 \
src/x_r1/grpo.py \
--config recipes/x_r1_test_sampling.yaml \
> ./output/test.log 2>&1
```

and we check log file: `./output/test.log`

## Q & A

### How to setting correct batch_Size and num_generations

we have 4gpu(1 vLLM + 3 training), setting config is:

```yaml
per_device_train_batch_size: 1
num_generations: 4
```

running with `--num_processes=3`: 

```text
ValueError: The global train batch size (3 x 1) must be evenly divisible by the number of generations per prompt (4). Given the current train batch size, the valid values for the number of generations are: [3].
```


( `per_device_train_batch_size` * `num_processes` ) % `num_generations` == 0


we should set

```yaml
# example 1
num_processes: 3
per_device_train_batch_size: 1
num_generations: 3
# 1 * 3 % 3 = 0

# example 2
num_processes: 3
per_device_train_batch_size: 4
num_generations: 6
# 4 * 3 % 6 = 0
```

if your have 8GPU(1vllm + 7training)

```yaml
num_processes: 7
per_device_train_batch_size: 4
num_generations: 14
# 4 * 7 % 14 = 0
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
