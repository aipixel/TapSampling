from __future__ import annotations

import argparse

from action_vae.common.config import TrainConfig
from action_vae.common.trainer import ActionVAETrainer


def parse_args() -> TrainConfig:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        description="Train the standalone action VAE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--h5_path", default=defaults.h5_path)
    parser.add_argument("--window_size", type=int, default=defaults.window_size)
    parser.add_argument("--num_workers", type=int, default=defaults.num_workers)
    parser.add_argument("--train_split_ratio", type=float, default=defaults.train_split_ratio)
    parser.add_argument("--batch_size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning_rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--grad_clip", type=float, default=defaults.grad_clip)
    parser.add_argument("--ema_decay", type=float, default=defaults.ema_decay)
    parser.add_argument("--use_amp", action="store_true", default=defaults.use_amp)
    parser.add_argument("--total_steps", type=int, default=defaults.total_steps)
    parser.add_argument("--weight_kl", type=float, default=defaults.weight_kl)
    parser.add_argument("--log_interval", type=int, default=defaults.log_interval)
    parser.add_argument("--val_interval", type=int, default=defaults.val_interval)
    parser.add_argument("--save_interval", type=int, default=defaults.save_interval)
    parser.add_argument("--resume_checkpoint", default=defaults.resume_checkpoint)
    parser.add_argument("--exp_name", default=defaults.exp_name)
    parser.add_argument("--save_dir", default=defaults.save_dir)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--num_layers", type=int, default=defaults.num_layers)
    parser.add_argument("--latent_tokens", type=int, default=defaults.latent_tokens)
    parser.add_argument("--latent_dim", type=int, default=defaults.latent_dim)
    namespace = parser.parse_args()
    return TrainConfig(
        h5_path=namespace.h5_path,
        window_size=namespace.window_size,
        num_workers=namespace.num_workers,
        train_split_ratio=namespace.train_split_ratio,
        batch_size=namespace.batch_size,
        learning_rate=namespace.learning_rate,
        grad_clip=namespace.grad_clip,
        ema_decay=namespace.ema_decay,
        use_amp=namespace.use_amp,
        total_steps=namespace.total_steps,
        weight_kl=namespace.weight_kl,
        log_interval=namespace.log_interval,
        val_interval=namespace.val_interval,
        save_interval=namespace.save_interval,
        resume_checkpoint=namespace.resume_checkpoint,
        exp_name=namespace.exp_name,
        save_dir=namespace.save_dir,
        seed=namespace.seed,
        device=namespace.device,
        num_layers=namespace.num_layers,
        latent_tokens=namespace.latent_tokens,
        latent_dim=namespace.latent_dim,
    )


def main() -> None:
    args = parse_args()
    trainer = ActionVAETrainer(args)
    try:
        trainer.train()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
