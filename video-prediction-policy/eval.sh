#!/usr/bin/env bash
set -euo pipefail

mkdir -p ./logs

# 'calvin_abc_dir' should be absolute path
nohup env CUDA_VISIBLE_DEVICES="0" \
    python -u policy_evaluation/evaluate_with_sampling.py \
    --video_model_path ./official_checkpoints/svd-robot-calvin-ft \
    --action_model_folder ./official_checkpoints/dp-calvin \
    --clip_model_path ./official_checkpoints/clip-vit-base-patch32 \
    --train_config VPP_Calvinabc_train \
    --seed 0 \
    --sample_type vae \
    --policy_sample_num 4 \
    --total_sample_num 24 \
    --max_batch_num 4 \
    --selection_type thres_weighted_merge \
    --selection_direction good \
    --selection_threshold 0.08 \
    --calvin_abc_dir /home/sizhe/clear_code/calvin/dataset/task_ABC_D \
    --verifier_calvin_path "../calvin" \
    --verifier_checkpoint "../tapsampling/official_chekpoint/last" \
    --eval_save_dir ./evaluation_logs/vae_4_24_weigth \
    > ../logs/vae_4_24_weigth.log 2>&1 &
