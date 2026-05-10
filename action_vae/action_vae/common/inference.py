from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
from torch.distributions import Independent, Normal

from action_vae.common.checkpoint import load_model_checkpoint
from action_vae.common.config import build_model_kwargs
from action_vae.common.runtime import resolve_device, seed_everything
from action_vae.model.mld_vae import AutoMldVae


class InferenceConfig(Protocol):
    checkpoint_path: str
    window_size: int
    feature_dim: int
    num_layers: int
    latent_tokens: int
    latent_dim: int
    seed: int
    device: str


class ActionVAEInference:
    def __init__(self, args: InferenceConfig) -> None:
        self.args = args
        seed_everything(args.seed, deterministic=True)
        self.device = resolve_device(args.device)

        self.feature_dim = args.feature_dim
        self.future_length = args.window_size
        self.history_length = 0
        self.window_size = self.history_length + self.future_length

        model_kwargs = build_model_kwargs(
            self.feature_dim,
            args.latent_tokens,
            args.latent_dim,
            args.num_layers,
        )
        self.model = AutoMldVae(**model_kwargs).to(self.device)
        if not args.checkpoint_path:
            raise ValueError("checkpoint_path is required")
        start_step = load_model_checkpoint(self.model, args.checkpoint_path, self.device)
        print(f"Loading checkpoint from {args.checkpoint_path} at step {start_step}")
        self.model.eval()

    def _as_action_tensor(self, actions: np.ndarray | torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(actions):
            tensor = actions.to(device=self.device, dtype=torch.float32)
        else:
            tensor = torch.as_tensor(actions, dtype=torch.float32, device=self.device)

        if tensor.ndim != 3:
            raise ValueError(f"Expected action tensor with shape (N, T, D), got {tuple(tensor.shape)}")
        if tensor.shape[1] != self.window_size:
            raise ValueError(f"Expected time dimension {self.window_size}, got {tensor.shape[1]}")
        if tensor.shape[2] != self.feature_dim:
            raise ValueError(f"Expected feature dimension {self.feature_dim}, got {tensor.shape[2]}")
        return tensor

    def encode_actions_into_dist(
        self,
        actions: np.ndarray | torch.Tensor,
        sample_actions: int = 0,
        replacement: bool = True,
        use_rsample: bool = False,
    ):
        with torch.no_grad():
            tensor = self._as_action_tensor(actions)
            future_motion_gt = tensor[:, -self.future_length :, :]
            history_motion = tensor[:, : self.history_length, :]
            _, dist = self.model.encode(future_motion=future_motion_gt, history_motion=history_motion)
            if sample_actions > 0:
                sampled_latents, _ = self.sample_latents_from_many(
                    dist,
                    sample_actions,
                    replacement=replacement,
                    use_rsample=use_rsample,
                )
                return dist, sampled_latents
            return dist, None

    def sample_latents_from_many(
        self,
        dist: Normal,
        sample_num: int,
        replacement: bool = True,
        use_rsample: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        loc = dist.loc
        scale = dist.scale
        if loc.dim() == 3 and loc.shape[0] == 1:
            loc = loc.squeeze(0)
            scale = scale.squeeze(0)
        elif loc.dim() != 2:
            raise ValueError(f"Unsupported loc shape: {tuple(loc.shape)}")

        k = loc.shape[0]
        if replacement:
            indices = torch.randint(0, k, (sample_num,), device=loc.device)
        else:
            indices = torch.randperm(k, device=loc.device)[:sample_num]

        sampled_dist = Independent(Normal(loc[indices], scale[indices]), 1)
        samples = sampled_dist.rsample() if use_rsample else sampled_dist.sample()
        return samples, indices

    def decode_latents_into_actions(
        self,
        latents: np.ndarray | torch.Tensor,
        history_motion: np.ndarray | torch.Tensor | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            if torch.is_tensor(latents):
                latent_tensor = latents.to(device=self.device, dtype=torch.float32)
            else:
                latent_tensor = torch.as_tensor(latents, dtype=torch.float32, device=self.device)

            if latent_tensor.ndim == 2:
                latent_tensor = latent_tensor.unsqueeze(0)
            if latent_tensor.ndim != 3:
                raise ValueError(f"Expected latent tensor with shape (L, N, D), got {tuple(latent_tensor.shape)}")

            batch_size = latent_tensor.shape[1]
            if history_motion is None:
                history_tensor = torch.empty(
                    batch_size,
                    self.history_length,
                    self.feature_dim,
                    dtype=latent_tensor.dtype,
                    device=latent_tensor.device,
                )
            else:
                history_tensor = self._as_action_tensor(history_motion)[:, : self.history_length, :]
            return self.model.decode(latent_tensor, history_tensor, nfuture=self.future_length)
