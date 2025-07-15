import argparse
import os
from safetensors import safe_open
import torch
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

# --- 1. 参数定义 ---
parser = argparse.ArgumentParser(
    description="Precisely convert custom LoRA weights to the PEFT format, handling special embedding layers."
)
parser.add_argument("--input_model", default="/data/jcxy/llm_model/PsyLLM-4.0-Turbo-09-13-E-I2", type=str, help="Path to the base Hugging Face model.")
parser.add_argument("--input_lora", default="/data/jcxy/haolu/workspace/lora/PsyLLM-4.0-Turbo-09-13-E-I2-Adult-Children-DPO/checkpoint-200-weight", type=str, help="Path to the directory containing custom LoRA weights (model.safetensors).")
parser.add_argument("--output_lora", default="/data/jcxy/haolu/workspace/lora/PsyLLM-4.0-Turbo-09-13-E-I2-Adult-Children-DPO/checkpoint-200-weight-peft", type=str, help="Path to save the converted LoRA in PEFT format.")
args = parser.parse_args()

# --- 2. 加载自定义 LoRA 权重 ---
print(f"Loading custom LoRA weights from: {args.input_lora}")
tensors = {}
lora_file_path = os.path.join(args.input_lora, "model.safetensors")
if not os.path.exists(lora_file_path):
    raise FileNotFoundError(f"LoRA weights file not found at: {lora_file_path}")

with safe_open(lora_file_path, framework="pt", device="cpu") as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)
print(f"Loaded {len(tensors)} tensors from custom LoRA file.")

# --- 3. 加载基础模型并应用 PEFT 配置 ---
print(f"Loading base model from: {args.input_model}")
model = AutoModelForCausalLM.from_pretrained(
    args.input_model,
    use_cache=False,
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
    device_map={"": "cpu"},
)

for param in model.parameters():
    param.requires_grad = False

target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj", "lm_head", "embed_tokens"]
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    r=128, 
    lora_alpha=256, 
    lora_dropout=0.0,
    bias="none",
    target_modules=target_modules
)

print("Applying PEFT configuration...")
model = get_peft_model(model, peft_config)

# --- 4. 核心逻辑：精确的权重映射 (修正版) ---
new_sd = model.state_dict()
mismatched_keys = []
mapped_keys_count = 0

# 动态确定基础前缀，以处理命名不一致问题
# 我们假设源和目标至少都有 'base_model.model'
source_base_prefix = "base_model.model."
if not any(k.startswith(source_base_prefix) for k in tensors.keys()):
    source_base_prefix = "" # 如果源没有这个前缀，就置空

peft_base_prefix = "base_model.model.model."
if not any(k.startswith(peft_base_prefix) for k in new_sd.keys()):
    # 如果 PEFT 模型的前缀不同，找到实际的公共前缀
    peft_base_prefix = os.path.commonprefix([k for k in new_sd.keys() if 'lora' in k]) or ""


print("\nStarting LoRA weight mapping process...")
for peft_key in new_sd:
    if 'lora' not in peft_key:
        continue
    
    # 从 PEFT key 中移除其独有的前缀和后缀，得到一个干净的、可识别的模块路径
    clean_peft_path = peft_key.replace(peft_base_prefix, "")
    clean_peft_path = clean_peft_path.replace('.default.weight', '').replace('.default', '')
    
    # 根据这个干净路径，构造我们期望在源文件中找到的键名
    source_key = None
    
    # --- 关键修复：专门处理 embed_tokens 的命名规则 ---
    if clean_peft_path.startswith('embed_tokens'):
        # 期望的源键没有 .weight 后缀
        source_key = source_base_prefix + clean_peft_path
            
    # 其他所有层，都应该有 .weight 后缀
    else:
        source_key = source_base_prefix + clean_peft_path + '.weight'
            
    # 加载权重或报告不匹配
    if source_key and source_key in tensors:
        print(f"  ✅ Mapping [Source] {source_key}  ->  [Target] {peft_key}")
        target_tensor = new_sd[peft_key]
        
        source_tensor = tensors[source_key]
        # 确保数据类型和设备匹配
        new_sd[peft_key] = source_tensor.to(target_tensor.device, target_tensor.dtype)
        
        # 检查并调整权重形状 (如果需要)
        if new_sd[peft_key].shape != source_tensor.shape:
             print(f"     ⚠️ Shape mismatch! Target: {new_sd[peft_key].shape}, Source: {source_tensor.shape}. This should not happen if r and alpha match.")

        mapped_keys_count += 1
    else:
        mismatched_keys.append((peft_key, source_key or f"Pattern not found for {peft_key}"))

# 加载 state_dict
model.load_state_dict(new_sd, strict=False)


# --- 5. 报告和保存 ---
print("\n" + "="*50)
print("LoRA Weight Mapping Report")
print("="*50)
total_lora_keys_in_model = len([k for k in new_sd if 'lora' in k])
print(f"Total LoRA parameters in PEFT model: {total_lora_keys_in_model}")
print(f"Successfully mapped parameters: {mapped_keys_count}")

if mismatched_keys:
    print(f"\n🚨 Unmapped parameters: {len(mismatched_keys)}")
    print("PEFT Key -> Tried to map from Source Key:")
    for peft_key, tried_key in mismatched_keys:
        print(f"  - {peft_key} -> {tried_key}")
else:
    print("\n✅ All LoRA parameters were successfully mapped!")

print("="*50)

if mapped_keys_count < total_lora_keys_in_model:
    print("\n\n⚠️ WARNING: Some LoRA weights were not mapped. Unmapped layers will have random initialization.")

print(f"\nSaving converted model to PEFT format at: {args.output_lora}")
os.makedirs(args.output_lora, exist_ok=True)
model.save_pretrained(args.output_lora)

adapter_config_path = os.path.join(args.output_lora, "adapter_config.json")
if not os.path.exists(adapter_config_path):
    peft_config.save_pretrained(args.output_lora)
    
print("\nConversion complete! ✨")