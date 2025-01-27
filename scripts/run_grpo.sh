
model_name_or_path=/mnt/sfs/shebin/models/Qwen/Qwen2.5-3B-Instruct
export HF_ENDPOINT=https://hf-mirror.com

accelerate launch --config_file configs/zero3.yaml src/open_r1/grpo.py \
    --output_dir output/DeepSeek-R1-Distill-Qwen-7B-GRPO \
    --model_name_or_path $model_name_or_path \
    --dataset_name AI-MO/NuminaMath-TIR \
    --max_prompt_length 256 \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --use_vllm True \
    --bf16