CUDA_VISIBLE_DEVICES=0 \
python -m action_vae.cli.train \
    --exp_name mvae_24 \
    --save_dir ./mvae_new \
    --h5_path ./actions.h5 \
    --window_size 10 \
    --total_steps 50000 \
    --val_interval 10000 \
    --save_interval 10000 \
    --batch_size 64 \
    --latent_tokens 1 \
    --latent_dim 24
