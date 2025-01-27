if [ -z "${NODE_RANK}" ] || [ -z "${NNODES}" ] || [ -z "${NUM_PROCESSES}" ] || [ -z "${MASTER_ADDR}" ]; then
    echo "Error: Required environment variables are not set"
    exit 1
fi

model_name_or_path=/mnt/sfs/shebin/models/Qwen/Qwen2.5-3B-Instruct
export HF_ENDPOINT=https://hf-mirror.com
old_config_yaml_file=configs/zero3.yaml
config_yaml_file=configs/zero3_machine_${NODE_RANK}.yaml

cp $old_config_yaml_file  $config_yaml_file
sed -i "s/num_machines: .*/num_machines: ${NNODES}/" $config_yaml_file
sed -i "s/num_processes: .*/num_processes: ${NUM_PROCESSES}/" $config_yaml_file
sed -i "s/machine_rank: .*/machine_rank: ${NODE_RANK}/" $config_yaml_file
sed -i "s#main_process_ip:.*#main_process_ip: \"${MASTER_ADDR}\"#" $config_yaml_file

accelerate launch --config_file $config_yaml_file src/open_r1/grpo.py \
    --output_dir output/DeepSeek-R1-Distill-Qwen-3B-GRPO \
    --model_name_or_path $model_name_or_path \
    --dataset_name AI-MO/NuminaMath-TIR \
    --max_prompt_length 256 \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --use_vllm False \
    --bf16