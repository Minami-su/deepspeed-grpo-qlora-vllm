
#!/bin/bash
OUTPUT_MODEL="/data2/jcxy/llm_model/Qwen2.5-0.5B"
original="/data2/jcxy/llm_model/Qwen2.5-0.5B"
#OUTPUT_MODEL="/data2/jcxy/llm_model/Qwen3-4B-AWQ"
OUTPUT_MODEL2="/data2/jcxy/llm_model/Qwen2.5-0.5B-X-R1-QloraV2"
#OUTPUT_MODEL2="/data2/jcxy/llm_model/Qwen3-4B-LIMO"
CHECKPOINT_DIR2="/data/jcxy/haolu/workspace/frameworks/X-R1/output/X-R1-test-QloraV2"

export NCCL_SOCKET_IFNAME=lo
CUDA_VISIBLE_DEVICES="2,3,4,5" ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 7832 \
    --config_file recipes/fsdp_qlora.yaml \
    --num_processes=3 src/x_r1/sgrpo.py \
    --config recipes/X_R1_zero_0dot5B_peft_config_4bit.yaml \
    --enable_lora=True \
    --model_name_or_path=${OUTPUT_MODEL} \
    --output_dir=${CHECKPOINT_DIR2} \
    > output/x_r1_0dotB_sampling.log 2>&1



# Automatically find the latest checkpoint folder based on modification time
LATEST_CHECKPOINT=$(ls -td ${CHECKPOINT_DIR2}/checkpoint-* | head -n 1)

# If no checkpoint is found, exit
if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "No checkpoint found in ${CHECKPOINT_DIR}"
    exit 1
fi

# Print the latest checkpoint folder for logging
echo "Using latest checkpoint: $LATEST_CHECKPOINT"

CUDA_VISIBLE_DEVICES=4 python /data/jcxy/haolu/workspace/training/alignment-handbook/scripts/merge_with_lora_lowmem-info.py \
    --input_model ${original} \
    --input_lora ${LATEST_CHECKPOINT}\
    --output_model ${OUTPUT_MODEL2}
