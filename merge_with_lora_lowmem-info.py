import json
import os
import torch
import peft
import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer,LlamaTokenizer
parser = argparse.ArgumentParser()
parser.add_argument("--output_model", default="", type=str)
parser.add_argument("--input_lora", default="", type=str)
parser.add_argument("--input_model", default="", type=str)

args = parser.parse_args()
input_lora = args.input_lora
ckpt=args.input_model
output_dir=args.output_model

tokenizer = AutoTokenizer.from_pretrained(ckpt,trust_remote_code=True)
# Original method without offloading
model = AutoModelForCausalLM.from_pretrained(
    ckpt,
    low_cpu_mem_usage=True,
    load_in_8bit=False,
    torch_dtype=torch.bfloat16,
    device_map={"": "cpu"},
)

from peft import PeftModel
model = PeftModel.from_pretrained(model, input_lora ,device_map={"": "cpu"},torch_dtype=torch.bfloat16)
model = model.merge_and_unload()
model._hf_peft_config_loaded = False
tokenizer.save_pretrained(output_dir)
print("Saving to Hugging Face format...")
model.save_pretrained(output_dir)
model.config.save_pretrained(output_dir)
