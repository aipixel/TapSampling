#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/deploy_action_vae_$(date +%F_%H-%M-%S).log"

CUDA_VISIBLE_DEVICES=0 \
nohup python -m action_vae.cli.serve \
    --checkpoint_path ./mvae/mvae_24_split/checkpoint_50000.pt \
    --window_size 10 \
    --feature_dim 7 \
    --num_layers 7 \
    --latent_tokens 1 \
    --latent_dim 24 \
    --host 0.0.0.0 \
    --port 8000 \
    > "$LOG_FILE" 2>&1 < /dev/null &

echo "started, pid=$!, log=$LOG_FILE"
