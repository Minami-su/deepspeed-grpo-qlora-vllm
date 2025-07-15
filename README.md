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
