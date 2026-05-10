from __future__ import annotations

from dataclasses import dataclass


DEFAULT_H5_PATH = "/home/sizhe/actions.h5"
DEFAULT_WINDOW_SIZE = 10
DEFAULT_FEATURE_DIM = 7
DEFAULT_ARCH = "all_encoder"
DEFAULT_NUM_LAYERS = 7
DEFAULT_LATENT_TOKENS = 1
DEFAULT_TRAIN_LATENT_DIM = 24
DEFAULT_INFERENCE_LATENT_DIM = 24
DEFAULT_H_DIM = 256
DEFAULT_FF_SIZE = 256
DEFAULT_NUM_HEADS = 4
DEFAULT_DROPOUT = 0.1
DEFAULT_POSITION_EMBEDDING = "learned"
DEFAULT_ACTIVATION = "gelu"


def build_model_kwargs(
    feature_dim: int,
    latent_tokens: int,
    latent_dim: int,
    num_layers: int,
) -> dict[str, object]:
    return {
        "nfeats": feature_dim,
        "latent_dim": (latent_tokens, latent_dim),
        "h_dim": DEFAULT_H_DIM,
        "ff_size": DEFAULT_FF_SIZE,
        "num_layers": num_layers,
        "num_heads": DEFAULT_NUM_HEADS,
        "dropout": DEFAULT_DROPOUT,
        "arch": DEFAULT_ARCH,
        "normalize_before": False,
        "activation": DEFAULT_ACTIVATION,
        "position_embedding": DEFAULT_POSITION_EMBEDDING,
    }


@dataclass
class TrainConfig:
    h5_path: str = DEFAULT_H5_PATH
    window_size: int = DEFAULT_WINDOW_SIZE
    num_workers: int = 4
    train_split_ratio: float = 0.8

    batch_size: int = 64
    learning_rate: float = 1e-4
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    use_amp: bool = False
    total_steps: int = 50000
    weight_kl: float = 1e-6
    log_interval: int = 100
    val_interval: int = 5000
    save_interval: int = 5000
    resume_checkpoint: str | None = None

    exp_name: str = "mvae_24"
    save_dir: str = "./mvae"
    seed: int = 0
    device: str = "cuda"

    num_layers: int = DEFAULT_NUM_LAYERS
    latent_tokens: int = DEFAULT_LATENT_TOKENS
    latent_dim: int = DEFAULT_TRAIN_LATENT_DIM


@dataclass
class ServeConfig:
    checkpoint_path: str = ""
    window_size: int = DEFAULT_WINDOW_SIZE
    feature_dim: int = DEFAULT_FEATURE_DIM
    num_layers: int = DEFAULT_NUM_LAYERS
    latent_tokens: int = DEFAULT_LATENT_TOKENS
    latent_dim: int = DEFAULT_INFERENCE_LATENT_DIM
    host: str = "0.0.0.0"
    port: int = 8000
    seed: int = 0
    device: str = "cuda"
