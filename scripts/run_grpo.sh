if [ -z "${NODE_RANK}" ] || [ -z "${NNODES}" ] || [ -z "${NUM_PROCESSES}" ] || [ -z "${MASTER_ADDR}" ]; then
    echo "Error: Required environment variables are not set"
    exit 1
fi

model_name_or_path=$MODEL_NAME_OR_PATH
if [ -z "$MODEL_NAME_OR_PATH" ]; then
  echo "Error: MODEL_NAME_OR_PATH environment variable is not set." >&2
  exit 1
fi

export HF_ENDPOINT=https://hf-mirror.com
old_config_yaml_file=configs/zero3.yaml
config_yaml_file=configs/zero3_machine_${MASTER_ADDR}_${NODE_RANK}.yaml

cp $old_config_yaml_file  $config_yaml_file
sed -i "s/num_machines: .*/num_machines: ${NNODES}/" $config_yaml_file
sed -i "s/num_processes: .*/num_processes: ${NUM_PROCESSES}/" $config_yaml_file
sed -i "s/machine_rank: .*/machine_rank: ${NODE_RANK}/" $config_yaml_file
sed -i "s#main_process_ip:.*#main_process_ip: \"${MASTER_ADDR}\"#" $config_yaml_file

#!/bin/bash

# 输入的环境变量
global_batch_size=${GLOBAL_BATCH_SIZE:-128}  # 默认全局批量大小为 32
num_process=${NUM_PROCESSES:-8}              # 默认进程数为 1

# 计算单卡 batch size
if [ "$num_process" -eq 0 ]; then
  echo "Error: num_process cannot be zero." >&2
  exit 1
fi

# 使用整数除法计算 single_batch_size，向下取整
single_batch_size=$((global_batch_size / num_process))

# 检查是否整除，如果有余数则加 1 向上取整
if [ $((global_batch_size % num_process)) -ne 0 ]; then
  single_batch_size=$((single_batch_size + 1))
fi

# 确保 single_batch_size 至少为 1
if [ "$single_batch_size" -lt 1 ]; then
  single_batch_size=1
fi

# 输出结果
echo "Single batch size: $single_batch_size"

env | grep HCCL
env | grep ATB
env | grep ASCEND
env | grep MODEL
env | grep NUM
env | grep NODE

last_two_model_name=$(basename $(dirname $MODEL_NAME_OR_PATH))/$(basename $MODEL_NAME_OR_PATH)
echo $last_two_model_name

accelerate launch --config_file $config_yaml_file src/open_r1/grpo.py \
    --output_dir output/${last_two_model_name}-GRPO \
    --model_name_or_path $model_name_or_path \
    --dataset_name AI-MO/NuminaMath-TIR \
    --max_prompt_length 256 \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps $single_batch_size \
    --logging_steps 1 \
    --use_vllm False \
    --bf16