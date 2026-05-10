from __future__ import annotations

from pathlib import Path

import torch


def load_model_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device) -> int:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_state_dict = checkpoint["model_state_dict"]
        start_step = int(checkpoint.get("num_steps", 0)) + 1
    else:
        model_state_dict = checkpoint
        start_step = 1

    if "latent_mean" not in model_state_dict:
        model_state_dict["latent_mean"] = torch.tensor(0.0)
    if "latent_std" not in model_state_dict:
        model_state_dict["latent_std"] = torch.tensor(1.0)

    model.load_state_dict(model_state_dict)
    return start_step


def save_model_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path, step: int) -> None:
    torch.save(
        {
            "num_steps": step,
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
