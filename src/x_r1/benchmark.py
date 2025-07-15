# CUDA_VISIBLE_DEVICES=0,1,2,3 python ./src/x_r1/benchmark.py \
# 	--model_name='xiaodongguaAIGC/X-R1-3B \
#   --dataset_name='HuggingFaceH4/MATH-500' \
# 	--output_name='./output/result_benchmark_math500'  \
# 	--max_output_tokens=1024 \
# 	--num_gpus=4

# CUDA_VISIBLE_DEVICES=0 python ./src/x_r1/benchmark.py \
# 	--model_name='xiaodongguaAIGC/X-R1-3B' \
#     --dataset_name='HuggingFaceH4/MATH-500' \
# 	--output_name='./output/result_benchmark_math500'  \
# 	--max_output_tokens=1024 \
# 	--num_gpus=1


from datasets import load_dataset, Dataset, DatasetDict
from vllm import LLM, SamplingParams
import argparse
import json
from grpo import SYSTEM_PROMPT
from rewards import accuracy_answer_reward
# import torch
import re
from transformers import AutoTokenizer 



def format_reward(completion):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    matches = re.match(pattern, completion)
    rewards = 1.0 if matches else 0.0 
    return rewards


def create_dataset(dataset_name, tokenizer):
    dataset = load_dataset(dataset_name, split='test')

    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    dataset = dataset.map(make_conversation)

    def format_function(example):
        example['prompt'] = tokenizer.apply_chat_template(example['prompt'], tokenize = False, add_generation_prompt = True )
        return example
    
    dataset = dataset.map(format_function, batched = False)
        
    return dataset


def vllm_generate(model_name, output_name, dataset_name, num_gpus, max_output_tokens):

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)

    # evaluation dataset
    dataset = create_dataset(dataset_name, tokenizer)
    print(dataset)


    answers = []
    prompts = []
    for data in dataset:
        answers.append(data['answer'])
        prompts.append(data['prompt'])

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.6,
                                     max_tokens=max_output_tokens,
                                     )
    # Create LLM object
    llm = LLM(model=model_name,  # replace your own model
                dtype='bfloat16',
                tensor_parallel_size=num_gpus,  # number of gpu
                gpu_memory_utilization=0.9,  # prevent OOM
                trust_remote_code=True,
                # use_cache=False,
              )

    # # vllm generation
    outputs = llm.generate(prompts,
                           sampling_params,)
    
    acc_scores = []
    format_scores = []
    result_all = []
    total_acc = 0
    total_format = 0
    for output, gold_answer, in zip (outputs, answers):
        prompt = output.prompt
        completion = output.outputs[0].text

        # print("Prompt: ", prompt)
        # print("completion:", completion)

        acc_score = accuracy_answer_reward(completion, gold_answer )
        acc_scores.append(acc_score)
        total_acc = total_acc + acc_score

        format_score = format_reward(completion)
        format_scores.append(format_score)
        total_format = total_format + format_score

        # print('format score', format_score)
        # print('accuracy score', acc_score)
        # print('-'*100)

        result_all.append({
            'prompt': prompt, 
            'completion': completion, 
            'gold answer': gold_answer, 
            'acc scores': acc_score,  
            'format score': format_score, 
        })

    print('='*100)
    print('eval acc: ', total_acc / len(acc_scores))
    print('eval format: ',total_format / len(format_scores))

    current_result_file = output_name + '.json'
    with open(current_result_file, 'w', encoding='utf-8') as file:
        json.dump(result_all, file, ensure_ascii=False, indent=4)
        
    return 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_name',  type=str, default='', required=True,
                        help='model name path')
    parser.add_argument('--output_name', type=str, default='', required=True,
                        help='output path')
    parser.add_argument('--dataset_name', type=str, default='HuggingFaceH4/MATH-500', required=True,
                        help='dataset path')
    parser.add_argument('--max_output_tokens', type=int, default=100,
                        help='generation tokens')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='generation tokens')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='generation tokens')

    args = parser.parse_args()
    print(args)

    vllm_generate(args.model_name,
                  args.output_name,
                  args.dataset_name,
                  args.num_gpus,
                  args.max_output_tokens,)
    # print(f'toxicity score mean: {mean}, toxicity score std: {std}')