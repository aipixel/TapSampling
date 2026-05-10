from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


OUTPUT_PATH = Path(__file__).resolve().parent / "actions.h5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="Path to a standard CALVIN split directory, for example /path/to/task_ABC_D/training",
    )
    return parser.parse_args()


def load_lang_indices(dataset_dir: Path) -> list[tuple[int, int]]:
    lang_ann_path = dataset_dir / "lang_annotations" / "auto_lang_ann.npy"
    lang_data = np.load(lang_ann_path, allow_pickle=True).item()
    return [(int(start_idx), int(end_idx)) for start_idx, end_idx in lang_data["info"]["indx"]]


def episode_path(dataset_dir: Path, episode_idx: int) -> Path:
    return dataset_dir / f"episode_{episode_idx:07d}.npz"


def load_actions(dataset_dir: Path, start_idx: int, end_idx: int) -> np.ndarray:
    actions = []
    for episode_idx in range(start_idx, end_idx):
        with np.load(episode_path(dataset_dir, episode_idx)) as episode:
            actions.append(episode["rel_actions"].astype(np.float32, copy=False))
    return np.stack(actions, axis=0)


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_path = OUTPUT_PATH

    ep_start_end_ids = load_lang_indices(dataset_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Dataset dir: {dataset_dir}")
    print(f"Output path: {output_path}")
    print(f"Found {len(ep_start_end_ids)} annotated segments")

    with h5py.File(output_path, "a") as handle:
        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            key_name = f"{start_idx}_{end_idx}"

            if key_name in handle:
                print(f"[SKIP] {key_name} already exists")
                continue

            actions = load_actions(dataset_dir, start_idx, end_idx)
            print(f"[{i + 1}/{len(ep_start_end_ids)}] {key_name} {actions.shape}")

            dataset = handle.create_dataset(
                key_name,
                data=actions,
                compression="gzip",
                compression_opts=4,
                chunks=True,
            )
            dataset.attrs["start_idx"] = start_idx
            dataset.attrs["end_idx"] = end_idx
            dataset.attrs["length"] = actions.shape[0]
            handle.flush()

    print("Finished writing actions.h5")


if __name__ == "__main__":
    main()
