#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_TIME="$(date +%Y%m%d_%H%M%S)"
CUDA_VISIBLE_DEVICES="0"
PORT="8002"
CALVIN_PATH="/home/sizhe/video-prediction-policy/calvin"
PRETRAINED_CHECKPOINT="/home/sizhe/video-prediction-policy/external_toolkits/VLA-Adapter/outputs_actiondim7/configs+calvin_abc+b16+lr-1e-05+lora-r64+dropout-0.0--image_aug--VLA-Adapter--calvin--actiondim7--65000_chkpt"
LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/verifier_server_${CURRENT_TIME}.log"

mkdir -p "$LOG_DIR"
cd "$SCRIPT_DIR"

nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PORT="$PORT" \
python vla-scripts/verifier_server.py \
--pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
--calvin_path "$CALVIN_PATH" \
> "$LOG_FILE" 2>&1 &

PID=$!
echo "deploy pid: $PID"
echo "deploy log: $LOG_FILE"

# env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PORT="$PORT" \
# python vla-scripts/verifier_server.py \
# --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
# --calvin_path "$CALVIN_PATH"
