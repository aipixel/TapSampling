#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_TIME="$(date +%Y%m%d_%H%M%S)"
DATA_NAME="calvin_abc"
CUDA_VISIBLE_DEVICES="0,1,2,3"
VLM_PATH="$SCRIPT_DIR/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b"
CONFIG_FILE_PATH="$SCRIPT_DIR/pretrained_models/configs"
DATA_ROOT_DIR="$SCRIPT_DIR/../calvin/dataset/task_ABC_D"
RUN_ROOT_DIR="$SCRIPT_DIR/outputs_actiondim7"
LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/taps_actiondim7--${CURRENT_TIME}.log"

mkdir -p "$RUN_ROOT_DIR" "$LOG_DIR"
cd "$SCRIPT_DIR"

nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" CALVIN_DATASET_ROOT="$DATA_ROOT_DIR" \
torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune_sp.py \
--vlm_path "$VLM_PATH" \
--config_file_path "$CONFIG_FILE_PATH" \
--dataset_name calvin_abc \
--use_film False \
--num_images_in_input 2 \
--use_proprio False \
--use_lora True \
--use_fz False \
--use_minivlm True \
--image_aug True \
--num_steps_before_decay 150000 \
--max_steps 150005 \
--save_freq 5000 \
--save_latest_checkpoint_only False \
--merge_lora_during_training True \
--batch_size 2 \
--grad_accumulation_steps 1 \
--learning_rate 1e-5 \
--lora_rank 8 \
--use_pro_version True \
--wandb_entity "YOUR_WANDB_ENTITY" \
--wandb_project "calvin_abc" \
--run_root_dir "$RUN_ROOT_DIR" \
--run_id_note "TAPS_actiondim7_${CURRENT_TIME}" \
> "$LOG_FILE" 2>&1 &

PID=$!
echo "train pid: $PID"
echo "train log: $LOG_FILE"
