"""
Minimal CALVIN success-percent dataset used by tapsampling training.

This local copy keeps only the logic exercised by `vla-scripts/finetune_sp.py`
and removes the broader `video-prediction-policy` dependency tree.
"""

import os
import pickle
import random
import re
from itertools import chain
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation as R, Slerp
from torch.utils.data import Dataset

TAPSAMPLING_ROOT = Path(__file__).resolve().parents[1]


def load_npz(filename: Path) -> Dict[str, np.ndarray]:
    return np.load(filename.as_posix())


def lookup_naming_pattern(dataset_dir: Path, save_format: str) -> Tuple[Tuple[Path, str], int]:
    iterator = os.scandir(dataset_dir)
    while True:
        filename = Path(next(iterator))
        if save_format in filename.suffix:
            break
    prefix_parts = re.split(r"\d+", filename.stem)
    naming_pattern = (filename.parent / prefix_parts[0], filename.suffix)
    n_digits = len(re.findall(r"\d+", filename.stem)[0])
    if len(naming_pattern) != 2 or n_digits <= 0:
        raise ValueError(f"Unexpected file naming pattern under {dataset_dir}")
    return naming_pattern, n_digits


def interpolate_action(values: np.ndarray, target_length: int) -> np.ndarray:
    source_steps = np.arange(values.shape[0])
    target_steps = np.linspace(0, values.shape[0] - 1, target_length)
    return np.interp(target_steps, source_steps, values)


def interpolate_binary_action(values: np.ndarray, target_length: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    source_steps = np.arange(values.shape[0], dtype=float)
    target_steps = np.linspace(0, values.shape[0] - 1, target_length)
    nearest_indices = np.rint(target_steps).astype(int)
    nearest_indices = np.clip(nearest_indices, 0, values.shape[0] - 1)
    return values[nearest_indices]


def interpolate_orient(eulers: np.ndarray, target_length: int) -> np.ndarray:
    source_times = np.linspace(0.0, 1.0, eulers.shape[0])
    rotations = R.from_euler("xyz", np.asarray(eulers, dtype=float), degrees=False)
    slerp = Slerp(source_times, rotations)
    target_times = np.linspace(source_times[0], source_times[-1], target_length)
    return slerp(target_times).as_euler("xyz", degrees=False)


def linear_percentage(start_value: float, end_value: float, num_steps: int) -> np.ndarray:
    return np.linspace(start_value, end_value, num_steps, dtype=float)


class ExtendedDiskDatasetSP(Dataset):
    def __init__(
        self,
        datasets_dir: Path,
        key: str,
        lang_folder: str,
        obs_space: DictConfig,
        batch_size: int = 32,
        min_window_size: int = 16,
        max_window_size: int = 32,
        num_workers: int = 1,
        skip_frames: int = 1,
        save_format: str = "npz",
        window_sampling_strategy: str = "geometric",
        geometric_p_value: float = 0.1,
        action_seq_len: int = 10,
        obs_seq_len: int = 1,
    ) -> None:
        if key != "lang":
            raise ValueError("ExtendedDiskDatasetSP only supports the language dataset path.")
        if save_format != "npz":
            raise ValueError("ExtendedDiskDatasetSP only supports npz episodes.")
        if window_sampling_strategy not in ("random", "geometric"):
            raise ValueError(f"Unsupported window sampling strategy: {window_sampling_strategy}")

        self.abs_datasets_dir = Path(datasets_dir)
        if not self.abs_datasets_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.abs_datasets_dir}")

        self.lang_folder = lang_folder
        self.obs_space = obs_space
        self.batch_size = batch_size
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.num_workers = num_workers
        self.skip_frames = skip_frames
        self.window_sampling_strategy = window_sampling_strategy
        self.geometric_p_value = geometric_p_value
        self.action_seq_len = action_seq_len
        self.obs_seq_len = obs_seq_len
        self.dataset_transform = None
        self.load_file = load_npz

        self.naming_pattern, self.n_digits = lookup_naming_pattern(self.abs_datasets_dir, save_format)
        (
            self.episode_lookup,
            self.lang_lookup,
            self.lang_ann,
            self.lang_text,
            self.success_percent,
            self.ep_start_end,
        ) = self._build_index()

    def __len__(self) -> int:
        return len(self.episode_lookup)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        episode = self._load_episode(idx)
        if self.dataset_transform is None:
            return episode
        return self.dataset_transform(episode)

    def _build_index(self) -> Tuple[np.ndarray, List[int], np.ndarray, np.ndarray, List[np.ndarray], np.ndarray]:
        lang_data_path = self.abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy"
        if not lang_data_path.exists():
            lang_data_path = self.abs_datasets_dir / "auto_lang_ann.npy"
        lang_data = np.load(lang_data_path, allow_pickle=True).item()

        ep_start_end_ids = np.asarray(lang_data["info"]["indx"])
        lang_ann = lang_data["language"]["emb"]
        lang_text = lang_data["language"]["ann"]

        with open(TAPSAMPLING_ROOT / "complete_percentage.pkl", "rb") as handle:
            success_percent = pickle.load(handle)

        episode_lookup: List[int] = []
        lang_lookup: List[int] = []
        for lang_idx, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            offset = 0
            for file_idx in range(start_idx, end_idx):
                if offset % self.skip_frames == 0:
                    lang_lookup.append(lang_idx)
                    episode_lookup.append(file_idx)
                offset += 1

        return np.array(episode_lookup), lang_lookup, lang_ann, lang_text, success_percent, ep_start_end_ids

    def _get_episode_name(self, file_idx: int) -> Path:
        return Path(f"{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}")

    def _load_episode(self, idx: int) -> Dict[str, np.ndarray]:
        start_idx = int(self.episode_lookup[idx])
        lang_idx = self.lang_lookup[idx]
        current_part_start, current_part_end = self.ep_start_end[lang_idx]
        part_success = self.success_percent[lang_idx]

        target_shift_length = random.randint(3, 12)
        part_shift = start_idx - current_part_start
        part_shift_right = current_part_end - start_idx

        if part_shift < target_shift_length:
            end_idx = start_idx + target_shift_length
        elif part_shift_right <= target_shift_length:
            end_idx = start_idx - target_shift_length
        else:
            end_idx = start_idx + random.choice((1, -1)) * target_shift_length

        direction = 1 if end_idx > start_idx else -1
        percent_range = (part_shift, end_idx - current_part_start)
        desired_action_steps = self.action_seq_len + self.obs_seq_len - 1

        if target_shift_length != desired_action_steps:
            success_percent = linear_percentage(
                part_success[percent_range[0]],
                part_success[percent_range[1]],
                desired_action_steps,
            )
        else:
            success_percent = part_success[percent_range[0] : percent_range[1] : direction]

        episodes = [
            self.load_file(self._get_episode_name(file_idx))
            for file_idx in range(start_idx, end_idx, direction)
        ]
        prev_episode = self.load_file(self._get_episode_name(start_idx - direction))

        episode = self._build_observation_dict(episodes)
        episode["language"] = self.lang_ann[lang_idx][0]
        episode["language_text"] = self.lang_text[lang_idx]
        episode["rel_actions"] = self._build_relative_actions(episode["actions"], prev_episode["actions"])
        episode["success_percent"] = success_percent
        return episode

    def _build_observation_dict(self, episodes: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        episode: Dict[str, np.ndarray] = {}
        keys = list(chain(self.obs_space["rgb_obs"], self.obs_space["state_obs"]))

        for key in keys:
            stacked = np.stack([sample[key] for sample in episodes])
            episode[key] = stacked[: self.obs_seq_len]

        scene_obs = np.stack([sample["scene_obs"] for sample in episodes])
        actions = np.stack([sample["actions"] for sample in episodes])
        episode["scene_obs"] = scene_obs[: self.obs_seq_len]
        episode["actions"] = actions[self.obs_seq_len - 1 :]
        return episode

    def _build_relative_actions(self, absolute_actions: np.ndarray, prev_action: np.ndarray) -> np.ndarray:
        if absolute_actions.shape[-1] < 7:
            raise ValueError(f"Expected absolute actions with at least 7 dims, got shape {absolute_actions.shape}")

        prev_and_current = np.concatenate([prev_action.reshape(1, -1), absolute_actions], axis=0)
        target_length = self.action_seq_len + self.obs_seq_len
        action_steps = target_length - 1

        interpolated_position = np.stack(
            [
                interpolate_action(prev_and_current[:, axis], target_length=target_length)
                for axis in range(3)
            ],
            axis=1,
        )
        rel_position = (interpolated_position[1:] - interpolated_position[:-1]) * 50

        interpolated_orientation = interpolate_orient(prev_and_current[:, 3:6], target_length)
        rel_orientation = (interpolated_orientation[1:] - interpolated_orientation[:-1]) * 20

        gripper = interpolate_binary_action(absolute_actions[:, 6], action_steps).reshape(-1, 1)

        return np.concatenate([rel_position, rel_orientation, gripper], axis=-1)
