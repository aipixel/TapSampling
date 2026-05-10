from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_REPOS = (
    "openai/clip-vit-base-patch32",
    "yjguo/svd-robot-calvin-ft",
    "yjguo/dp-calvin",
)


def download_repo(repo_id: str, output_root: Path) -> Path:
    repo_name = repo_id.rsplit("/", maxsplit=1)[-1]
    local_dir = output_root / repo_name
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {repo_id} to {local_dir}...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        token=os.environ.get("HF_TOKEN"),
    )
    return local_dir


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    output_root = script_dir / "official_checkpoints"
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Saving checkpoints under {output_root}")
    for repo_id in MODEL_REPOS:
        download_repo(repo_id, output_root)

    print("Finished downloading official checkpoints.")


if __name__ == "__main__":
    main()
