import argparse
import pickle
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "complete_percentage.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        type=Path,
        required=True,
        help="Dataset root directory, for example /path/to/calvin/dataset/task_ABC_D",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output pkl path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    return parser.parse_args()


def resolve_auto_lang_ann_path(dataset_root: Path) -> Path:
    candidates = [
        dataset_root / "training" / "lang_clip_resnet50" / "auto_lang_ann.npy",
        dataset_root / "training" / "auto_lang_ann.npy",
        dataset_root / "auto_lang_ann.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"auto_lang_ann.npy not found under {dataset_root}. Tried: {candidates}")


def build_complete_percentage(auto_lang_ann_path: Path) -> list[np.ndarray]:
    data = np.load(auto_lang_ann_path, allow_pickle=True).item()

    inds = list(data["info"]["indx"])
    starts = np.array([t[0] for t in inds], dtype=int)
    order = np.argsort(starts)

    ann = np.array(data["language"]["ann"], dtype=object)
    task = np.array(data["language"]["task"], dtype=object)

    inds_sorted = [inds[i] for i in order]
    ann_sorted = list(ann[order])
    task_sorted = list(task[order])

    data["info"]["indx"] = inds_sorted
    data["language"]["ann"] = ann_sorted
    data["language"]["task"] = task_sorted

    sorted_percent = []
    current_idx = 0
    while True:
        if current_idx >= len(inds_sorted):
            break

        _, end = inds_sorted[current_idx]
        task_name = task_sorted[current_idx]
        end_idx = current_idx

        for search_idx in range(current_idx + 1, 99999999):
            if search_idx >= len(inds_sorted):
                break
            if task_sorted[search_idx] != task_name:
                break
            if inds_sorted[search_idx][0] > end:
                break
            end = max(end, inds_sorted[search_idx][1])
            end_idx = search_idx

        total_sequence = [inds_sorted[episode_part] for episode_part in range(current_idx, end_idx + 1)]
        true_start = np.min(total_sequence)
        true_end = np.max(total_sequence)
        total_len = true_end - true_start

        for episode_part in range(current_idx, end_idx + 1):
            part_success_percent = (
                np.array(range(inds_sorted[episode_part][0], inds_sorted[episode_part][1])) - true_start
            ) / (total_len - 1)
            sorted_percent.append(part_success_percent)

        current_idx = end_idx + 1

    inv = np.argsort(order)
    return [sorted_percent[w] for w in inv]


def main() -> None:
    args = parse_args()
    auto_lang_ann_path = resolve_auto_lang_ann_path(args.dataset_root)
    complete_percentage = build_complete_percentage(auto_lang_ann_path)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("wb") as handle:
        pickle.dump(complete_percentage, handle)

    print(f"Loaded annotations from: {auto_lang_ann_path}")
    print(f"Saved {len(complete_percentage)} entries to: {args.output_path}")


if __name__ == "__main__":
    main()
