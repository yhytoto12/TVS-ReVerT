#!/bin/bash

SCRIPT_NAME="train/train_sff.py"
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG="Qwen2.5-7B"
EXP_NAME="SFF:${MODEL_TAG}"
OUTPUT_DIR="outputs/${EXP_NAME}"
MAX_LENGTH=1600

# Command bash train_sff.sh -g <gpus>
# -g: number of gpus

N_GPUS=1

while getopts "g:" opt; do
  case $opt in
    g) N_GPUS=$OPTARG ;;
    *) echo "Invalid option: -$OPTARG" >&2 ;;
  esac
done

echo "============================================================"
if [ "$N_GPUS" -eq 1 ]; then
    echo "Run Single-GPU training"
else
    echo "Run Multi-GPU training with accelerate on $N_GPUS GPUs"
fi

echo "Experiment name: ${EXP_NAME}"
echo "Model name: ${MODEL_NAME}"
echo "Output directory: ${OUTPUT_DIR}"
echo "==========================================================="

TOTAL_BATCH_SIZE=8
GRADIENT_ACC_STEPS=$((TOTAL_BATCH_SIZE / N_GPUS))

if [ $N_GPUS -eq 1 ]; then
    # Single GPU training -> use vanilla python
    python $SCRIPT_NAME \
        --output_dir "${OUTPUT_DIR}" \
        --log_level debug \
        --bf16 True \
        --model_name_or_path "${MODEL_NAME}" \
        --num_train_epochs 1 \
        --dataset_name "all" \
        --max_length ${MAX_LENGTH} \
        --use_lora True \
        --learning_rate 2e-5 \
        --adam_beta1 0.9 \
        --adam_beta2 0.999 \
        --weight_decay 0.1 \
        --warmup_ratio 0.1 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps ${GRADIENT_ACC_STEPS} \
        --save_strategy "steps" \
        --save_steps 500 \
        --logging_steps 10 \
        --optim "paged_adamw_8bit" \
        --lr_scheduler_type "cosine" \
        --overwrite_output_dir

else
    # Multi-GPU training -> use accelerate
    accelerate launch \
        --num_processes=$N_GPUS \
        --num_machines=1 \
        --main_process_port=12345 \
        --config_file scripts/deepspeed_config.yaml \
    $SCRIPT_NAME \
        --output_dir "${OUTPUT_DIR}" \
        --log_level debug \
        --bf16 True \
        --model_name_or_path "${MODEL_NAME}" \
        --num_train_epochs 1 \
        --dataset_name "all" \
        --max_length ${MAX_LENGTH} \
        --use_lora True \
        --learning_rate 2e-5 \
        --adam_beta1 0.9 \
        --adam_beta2 0.999 \
        --weight_decay 0.1 \
        --warmup_ratio 0.1 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps ${GRADIENT_ACC_STEPS} \
        --save_strategy "steps" \
        --save_steps 500 \
        --logging_steps 10 \
        --optim "paged_adamw_8bit" \
        --lr_scheduler_type "cosine" \
        --overwrite_output_dir
fi