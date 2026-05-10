from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download


@dataclass(frozen=True)
class DownloadSpec:
    repo_id: str
    local_dir_name: str
    allow_patterns: tuple[str, ...]


MODEL_SPECS = (
    DownloadSpec(
        repo_id="Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b",
        local_dir_name="prism-qwen25-extra-dinosiglip-224px-0_5b",
        allow_patterns=(
            "README.md",
            "config.json",
            "config.yaml",
            "prism-qwen25-extra-dinosiglip-224px+0_5b+stage-finetune+x7.jsonl",
            "run-metrics.jsonl",
            "checkpoints/step-020792-epoch-01-loss=0.5268.pt",
        ),
    ),
    DownloadSpec(
        repo_id="Qwen/Qwen2.5-0.5B",
        local_dir_name="Qwen2.5-0.5B",
        allow_patterns=(
            "LICENSE",
            "README.md",
            "config.json",
            "generation_config.json",
            "merges.txt",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        ),
    ),
    DownloadSpec(
        repo_id="timm/vit_large_patch14_reg4_dinov2.lvd142m",
        local_dir_name="vit_large_patch14_reg4_dinov2.lvd142m",
        allow_patterns=(
            "README.md",
            "config.json",
            "model.safetensors",
            "pytorch_model.bin",
        ),
    ),
    DownloadSpec(
        repo_id="timm/ViT-SO400M-14-SigLIP",
        local_dir_name="ViT-SO400M-14-SigLip",
        allow_patterns=(
            "open_clip_pytorch_model.bin",
        ),
    ),
)


def _snapshot_download(repo_id: str, local_dir: Path, allow_patterns: tuple[str, ...]) -> None:
    kwargs = {
        "repo_id": repo_id,
        "local_dir": str(local_dir),
        "allow_patterns": list(allow_patterns),
        "token": os.environ.get("HF_TOKEN"),
    }

    # Older huggingface_hub versions still support disabling symlink-based local downloads.
    if "local_dir_use_symlinks" in inspect.signature(snapshot_download).parameters:
        kwargs["local_dir_use_symlinks"] = False

    snapshot_download(**kwargs)


def download_repo(spec: DownloadSpec, output_root: Path) -> Path:
    local_dir = output_root / spec.local_dir_name
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {spec.repo_id} to {local_dir}...")
    _snapshot_download(spec.repo_id, local_dir, spec.allow_patterns)
    return local_dir


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    output_root = script_dir / "pretrained_models"
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Saving base model checkpoints under {output_root}")
    for spec in MODEL_SPECS:
        download_repo(spec, output_root)

    print("Finished downloading base model checkpoints.")


if __name__ == "__main__":
    main()
