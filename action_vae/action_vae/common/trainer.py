from __future__ import annotations

import copy
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from action_vae.common.checkpoint import load_model_checkpoint, save_model_checkpoint
from action_vae.common.config import TrainConfig, build_model_kwargs
from action_vae.common.runtime import resolve_device, seed_everything
from action_vae.data.simple_action_dataset import ActionDataset
from action_vae.model.mld_vae import AutoMldVae


class ActionVAETrainer:
    def __init__(self, args: TrainConfig) -> None:
        self.args = args
        seed_everything(args.seed, deterministic=True)

        self.device = resolve_device(args.device)
        self.use_amp = bool(args.use_amp) and self.device.type == "cuda"
        self.scaler = amp.GradScaler(enabled=self.use_amp)

        save_root = Path(args.save_dir)
        self.save_dir = save_root / args.exp_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        train_dataset = ActionDataset(
            h5_path=args.h5_path,
            window_size=args.window_size,
            split="train",
            train_split_ratio=args.train_split_ratio,
        )
        val_dataset = ActionDataset(
            h5_path=args.h5_path,
            window_size=args.window_size,
            split="val",
            train_split_ratio=args.train_split_ratio,
        )

        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty after windowing; check h5_path and window_size")

        self.history_length = 0
        self.future_length = args.window_size
        self.feature_dim = train_dataset.feature_dim

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=self.device.type == "cuda",
        )

        model_kwargs = build_model_kwargs(
            self.feature_dim,
            args.latent_tokens,
            args.latent_dim,
            args.num_layers,
        )
        self.model = AutoMldVae(**model_kwargs).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.learning_rate)
        self.model_avg = copy.deepcopy(self.model) if args.ema_decay > 0 else None
        if self.model_avg is not None:
            self.model_avg.eval()

        self.start_step = 1
        if args.resume_checkpoint:
            self.start_step = load_model_checkpoint(self.model, args.resume_checkpoint, self.device)
            if self.model_avg is not None:
                self.model_avg.load_state_dict(self.model.state_dict())
            print(f"Loading checkpoint from {args.resume_checkpoint} at step {self.start_step}")

        with open(self.save_dir / "config.yaml", "w", encoding="utf-8") as handle:
            yaml.safe_dump(asdict(args), handle, sort_keys=False)

        # run_name = f"{args.exp_name}__seed{args.seed}__{int(time.time())}"
        # from torch.utils.tensorboard import SummaryWriter
        # self.writer = SummaryWriter(log_dir=str(self.save_dir / "tensorboard" / run_name))
        # self.writer.add_text("config", f"```yaml\n{yaml.safe_dump(asdict(args), sort_keys=False)}\n```")

        self.rec_criterion = torch.nn.HuberLoss(reduction="mean", delta=1.0)
        self.step = self.start_step
        print("model config:", model_kwargs)
        print("using device:", self.device)

    def calc_loss(
        self,
        future_motion_gt: torch.Tensor,
        future_motion_pred: torch.Tensor,
        dist: torch.distributions.Normal,
    ) -> dict[str, torch.Tensor]:
        mu_ref = torch.zeros_like(dist.loc)
        scale_ref = torch.ones_like(dist.scale)
        dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
        kl_loss = torch.distributions.kl_divergence(dist, dist_ref).mean()
        rec_loss = self.rec_criterion(future_motion_pred, future_motion_gt)
        loss = self.args.weight_kl * kl_loss + rec_loss
        return {
            "kl_loss": kl_loss,
            "rec_loss": rec_loss,
            "loss": loss,
        }

    def train(self) -> None:
        total_steps = self.args.total_steps
        if total_steps <= 0:
            raise ValueError("Total training steps must be positive")

        if self.start_step > total_steps:
            print(f"Checkpoint step {self.start_step} is already beyond total_steps={total_steps}; nothing to do.")
            return

        self.model.train()
        progress = tqdm(total=total_steps - self.start_step + 1)
        loss_window: list[float] = []

        for epoch_idx in range(999999):
            for batch in self.train_loader:
                if self.step > total_steps:
                    break

                frac = 1.0 - (self.step - 1.0) / total_steps
                self.optimizer.param_groups[0]["lr"] = frac * self.args.learning_rate

                motion = batch["actions"].to(self.device, non_blocking=self.device.type == "cuda")
                future_motion_gt = motion[:, -self.future_length :, :]
                history_motion = motion[:, : self.history_length, :]

                self.optimizer.zero_grad(set_to_none=True)
                with amp.autocast(enabled=self.use_amp):
                    latent, dist = self.model.encode(
                        future_motion=future_motion_gt,
                        history_motion=history_motion,
                    )
                    future_motion_pred = self.model.decode(
                        latent,
                        history_motion,
                        nfuture=self.future_length,
                    )
                    loss_dict = self.calc_loss(future_motion_gt, future_motion_pred, dist)

                loss = loss_dict["loss"]
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.model_avg is not None:
                    for param, avg_param in zip(self.model.parameters(), self.model_avg.parameters()):
                        avg_param.data.mul_(self.args.ema_decay).add_(
                            param.data,
                            alpha=1 - self.args.ema_decay,
                        )

                self._maybe_log(loss_dict)
                self._maybe_save(total_steps)
                self._maybe_validate(total_steps)

                loss_window.append(float(loss.item()))
                loss_window = loss_window[-100:]
                progress.set_postfix(epoch=epoch_idx, loss=f"{np.mean(loss_window):.4f}")
                progress.update(1)
                self.step += 1

            if self.step > total_steps:
                break

    def _maybe_log(self, loss_dict: dict[str, torch.Tensor]) -> None:
        if self.args.log_interval <= 0:
            return
        if self.step % self.args.log_interval != 0:
            return

        # print(f"step {self.step}, save to log")
        # for key, value in loss_dict.items():
        #     self.writer.add_scalar(f"loss/{key}", value.item(), self.step)
        # self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.step)

    def _maybe_save(self, total_steps: int) -> None:
        if self.args.save_interval <= 0:
            return
        if self.step % self.args.save_interval != 0 and self.step != total_steps:
            return
        self.save()

    def _maybe_validate(self, total_steps: int) -> None:
        if self.args.val_interval <= 0:
            return
        if self.step % self.args.val_interval != 0 and self.step != total_steps:
            return
        self.validate()

    def validate(self) -> None:
        if len(self.val_loader) == 0:
            print("validation skipped: empty val dataset")
            return

        original_mode = self.model.training
        self.model.eval()
        losses_dict: dict[str, list[torch.Tensor]] = {}

        with torch.no_grad():
            for batch in tqdm(self.val_loader, leave=False):
                motion = batch["actions"].to(self.device, non_blocking=self.device.type == "cuda")
                future_motion_gt = motion[:, -self.future_length :, :]
                history_motion = motion[:, : self.history_length, :]

                latent, dist = self.model.encode(future_motion=future_motion_gt, history_motion=history_motion)
                future_motion_pred = self.model.decode(
                    latent,
                    history_motion,
                    nfuture=self.future_length,
                )
                loss_dict = self.calc_loss(future_motion_gt, future_motion_pred, dist)
                for key, value in loss_dict.items():
                    losses_dict.setdefault(key, []).append(value.detach())

        for key, values in losses_dict.items():
            mean_value = torch.stack(values).mean().item()
            print(key, mean_value)
            # self.writer.add_scalar(f"val_loss/{key}", mean_value, self.step)

        self.model.train(original_mode)

    def save(self) -> None:
        model_to_save = self.model if self.model_avg is None else self.model_avg
        checkpoint_path = self.save_dir / f"checkpoint_{self.step}.pt"
        save_model_checkpoint(model_to_save, checkpoint_path, self.step)
        print(f"Saved checkpoint at {checkpoint_path}")

    def close(self) -> None:
        # self.writer.close()
        pass
