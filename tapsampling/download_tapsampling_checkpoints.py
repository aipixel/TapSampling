from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download


@dataclass(frozen=True)
class DownloadSpec:
    repo_id: str
    output_dir: str


DOWNLOAD_SPECS = (
    DownloadSpec(
        repo_id="SizheZhao/TapSampling-Action-VAE",
        output_dir="../action_vae",
    ),
    DownloadSpec(
        repo_id="SizheZhao/TapSampling",
        output_dir="official_checkpoint",
    ),
)


def _snapshot_download(spec: DownloadSpec, local_dir: Path) -> None:
    kwargs = {
        "repo_id": spec.repo_id,
        "local_dir": str(local_dir),
        "token": os.environ.get("HF_TOKEN"),
    }

    # Older huggingface_hub versions still support disabling symlink-based local downloads.
    if "local_dir_use_symlinks" in inspect.signature(snapshot_download).parameters:
        kwargs["local_dir_use_symlinks"] = False

    snapshot_download(**kwargs)


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    for spec in DOWNLOAD_SPECS:
        output_dir = (script_dir / spec.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading TapSampling checkpoints from {spec.repo_id} to {output_dir}")
        _snapshot_download(spec, output_dir)

    print("Finished downloading TapSampling checkpoints.")


if __name__ == "__main__":
    main()
