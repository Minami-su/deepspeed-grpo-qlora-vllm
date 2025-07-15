# X-R1

![x-r1-logo](./README.assets/X-R1-log.png)


X-R1 aims to build an easy-to-use, low-cost training framework based on end-to-end reinforcement learning to accelerate the development of Scaling Post-Training

Inspired by [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) and [open-r1](https://github.com/huggingface/open-r1) , we produce minimal-cost for training 0.5B R1-Zero "Aha Moment"💡 from base model


## Feature

- 🔥Training with LoRA
- 4x3090/4090 GPUs training 1hour, 💰cost < 7 dollar, 10min 37'step output “aha Moment“ 💡
- 0.5B scale model RL training
- support BIGGER model: 1.5B/7B/32B...
- We supply 0.75k/1.5k/7.5k dataset for fast train loop
- We logging GRPO online sampling data to log file


## News

- 2025.02.18 Suppor LoRA+Zero3, Medical and llm-as-a-reward, add MATH500 benchmark evaluation result.
- 2025.02.16 Support LoRA
- 2025.02.15 Release Chinese Training
- 2025.02.13 Release X-R1-3B, whick better follow format. colab inference
- 2025.02.12 Release X-R1-1.5B config/wandb/model/log
- 2025.02.12: Release X-R1 first version

## Result

### Overview

Running Scripts:

```bash
bash ./scripts/run_x_r1_zero.sh
```

We would share training details about  config/wandb/model/log, also evaluation results:

📈 [wandb details](https://api.wandb.ai/links/xiaodonggua/eb471rlw) | 🔥 [Colab Inference](https://colab.research.google.com/drive/1TxjJ-M9J2lLW3zcKr7oeER3snXe0oWo4#scrollTo=VnkmSMGwZOhI) | 🤗 [Models](https://huggingface.co/xiaodongguaAIGC)

We have confirmed the effectiveness of the X-R1 RL-Zero-training method for `0.5B/1.5B/3B-Base` model, We can observe that in the without-SFT, reinforcement learning has **Incentivizing** the model's reasoning abilities and format-following capabilities, and the experimental results of X-R1 are very encouraging.

![X-R1-base-result-curves](./README.assets/X-R1-base-result-curves.png)

training config

| Model                 | 0.5B                                                                                         | 1.5B                                                                                         | 3B                                                                                           | 7B  |
| --------------------- | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | --- |
| TargetModel           | [X-R1-0.5B](https://huggingface.co/xiaodongguaAIGC/X-R1-0.5B)                                | [X-R1-1.5B](https://huggingface.co/xiaodongguaAIGC/X-R1-1.5B)                                | [X-R1-3B](https://huggingface.co/xiaodongguaAIGC/X-R1-3B)                                    |     |
| Log                   | [[link]](https://drive.google.com/file/d/1m-w0B2L9o-bwGDgaOtWFLR0C0MAEBTFQ/view?usp=sharing) | [[link]](https://drive.google.com/file/d/11tBShY206Pu_SxWE0M-mG2_Cdf9mFNig/view?usp=sharing) | [[link]](https://drive.google.com/file/d/1t4WzsK0aMrULYKjKsKH29LsWQMeTDjTb/view?usp=sharing) |     |
| GPU                   | 4x3090                                                                                       | 4x3090                                                                                       | 4x3090                                                                                       |     |
| Base                  | Qwen/Qwen2.5-0.5B                                                                            | Qwen/Qwen2.5-1.5B                                                                            | Qwen/Qwen2.5-3B                                                                              |     |
| Dataset               | X-R1-750                                                                                     | X-R1-750                                                                                     | X-R1-750                                                                                     |     |
| Config: recipes       | X_R1_zero_0dot5B_config.yaml                                                                 | X_R1_zero_1dot5B_config.yaml                                                                 | X_R1_zero_3B_config.yaml                                                                     |     |
| num_generations       | 16                                                                                           | 8                                                                                            | 4                                                                                            |     |
| max_completion_length | 1024                                                                                         | 1024                                                                                         | 1024                                                                                         |     |
| num_train_epochs      | 3                                                                                            | 3                                                                                            | 3                                                                                            |     |
| Times                 | 1:14:10                                                                                      | 1:59:06                                                                                      | 2:23:06                                                                                      |     |

### Example: 0.5B R1-Zero

0.5B, 4x3090.  if you have 4 GPUs, you should set `--num_processes=3`.  One GPU deploy vLLM as online inference engine, for faster GRPO sampling

example: 4x4090, 3epochs, training time, ~1h20min

```shell
ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=3 \
src/x_r1/grpo.py \
--config recipes/X_R1_zero_0dot5B_config.yaml \
> ./output/x_r1_0dotB_sampling.log 2>&1
```

tips : use `--config recipes/X_R1_zero_3B_config.yaml` for better learning reasoning and format

#### Aha Moment:

***Wait**, that doesn't match either of our options. It seems like I made a **mistake** in my **assumptions**. **Let's go back** to the original equations*

![aha_moment](./README.assets/aha_moment_0.5B.png)

#### benchmark evaluation

we use vllm as backend, to evaluate benchmark. output accuracy-metric and format-metric, and json file 

```bash
CUDA_VISIBLE_DEVICES=0,1 python ./src/x_r1/benchmark.py \
	--model_name='xiaodongguaAIGC/X-R1-0.5B' \
    --dataset_name='HuggingFaceH4/MATH-500' \
	--output_name='./output/result_benchmark_math500'  \
	--max_output_tokens=1024 \
	--num_gpus=2
```

### Example: Chinese Math Reasoning

X-R1 support chinese math reasoning, it's easy to make chinese `Aha Moment`, as follow

```bash
ACCELERATE_LOG_LEVEL=info \
accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=3 \
src/x_r1/grpo.py \
--config recipes/examples/mathcn_zero_3B_config.yaml \
> ./output/mathcn_3B_sampling.log 2>&1
```

#### reward curve

X-R1 use 4x3090 ~16h training 3B-base with 7.5k chinese math problem.

![X-R1-math-cn-curve](./README.assets/X-R1-math-cn-curve.png)

#### Chinese Aha Moment

[X-R1-3B-CN](xiaodongguaAIGC/X-R1-0.5B-CN) training [log](https://drive.google.com/file/d/1dPex_uiZ-4Lj2Jv8G8SWw6z0OsNSqLLM/view?usp=sharing) we track ”Aha Moment”

![X-R1-Math-cn-AhaMoment-1](./README.assets/X-R1-Math-cn-AhaMoment-1.png)

![X-R1-Math-cn-AhaMoment-2](./README.assets/X-R1-Math-cn-AhaMoment-2.png)



### Example: GRPO + LoRA

1. multi-gpu run:

```bash
ACCELERATE_LOG_LEVEL=info 
accelerate launch 
--config_file recipes/zero3.yaml 
--num_processes=3 src/x_r1/grpo.py 
--config recipes/examples/X_R1_zero_7B_peft_usevllm_config.yaml.yaml
> ./output/test_7b_lora_sampling.log 2>&1
```

2. single-gpu 3090 training 7B LoRA run:

```bash
ACCELERATE_LOG_LEVEL=info 
accelerate launch 
--config_file recipes/zero3.yaml 
--num_processes=1 src/x_r1/grpo.py 
--config recipes/examples/X_R1_zero_7B_peft_novllm_config.yaml.yaml
> ./output/test_7b_lora_sampling.log 2>&1
```

### Example: GRPO without KL

set KL iterm `beta: 0.0` and ignore `ref_model` , improve 20% performance.

```bash 
accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=3 \
src/x_r1/grpo.py \
--config recipes/X_R1_zero_1dot5B_noKL_config.yaml \
> ./output/test_1dot5B_sampling.log 2>&1
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

for test environment:

```bash
mkdir output
```

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

- support QLoRA GRPO Trainning
- Release 7B config/result
- add more rule reward
- support more base model
- add benchmark evaluation reuslt

## About

If you have any suggestions, please contact: dhcode95@gmail.com

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
