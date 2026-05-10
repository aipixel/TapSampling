import os.path
from itertools import chain
import logging
import torch
from pathlib import Path
import pickle
from typing import Any, Dict, List, Tuple
import random

from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
import numpy as np

from policy_models.datasets.utils.episode_utils import lookup_naming_pattern
import json
logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAR_CODE_ROOT = PROJECT_ROOT.parent
TAPSAMPLING_ROOT = CLEAR_CODE_ROOT / "tapsampling"


def load_pkl(filename: Path) -> Dict[str, np.ndarray]:
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_npz(filename: Path) -> Dict[str, np.ndarray]:
    return np.load(filename.as_posix())

from policy_models.datasets.utils.episode_utils import (
    get_state_info_dict,
    process_actions,
    process_language,
    process_rgb,
    process_state,
)

# class FNV1_32_Hasher:
#     def __call__(self, data: str) -> int:
#         fnv_prime = 0x01000193
#         hash_val = 0x811c9dc5

#         for byte in data.encode('utf-8'):
#             hash_val = (hash_val * fnv_prime) & 0xffffffff
#             hash_val = hash_val ^ byte

#         return hash_val

import pyhash

# hasher = FNV1_32_Hasher()
hasher = pyhash.fnv1_32()
logger = logging.getLogger(__name__)

def interpolate_action(arr, target_length: int) -> np.ndarray:
    N = arr.shape[0]
    x = np.linspace(0, N - 1, target_length)
    xp = np.arange(N)
    return np.interp(x, xp, arr)

from scipy.spatial.transform import Rotation as R, Slerp
def interpolate_orient(eulers: np.ndarray, M: int, *,
                       seq_order: str = "xyz",
                       degrees: bool = False) -> np.ndarray:
    eulers = np.asarray(eulers, dtype=float)
    N = eulers.shape[0]
    times = np.linspace(0.0, 1.0, N)
    # 构造 Rotation 对象
    rots = R.from_euler(seq_order, eulers, degrees=degrees)
    # Slerp 插值器
    slerp = Slerp(times, rots)

    target_times = np.linspace(times[0], times[-1], M)
    interp_rots = slerp(target_times)
    interp_eulers = interp_rots.as_euler(seq_order, degrees=degrees)
    return interp_eulers



def get_validation_window_size(idx: int, min_window_size: int, max_window_size: int) -> int:
    """
    In validation step, use hash function instead of random sampling for consistent window sizes across epochs.

    Args:
        idx: Sequence index.
        min_window_size: Minimum window size.
        max_window_size: Maximum window size.

    Returns:
        Window size computed with hash function.
    """
    window_range = max_window_size - min_window_size + 1
    return min_window_size + hasher(str(idx)) % window_range

from omegaconf import DictConfig
from torch.utils.data import Dataset
from typing import Dict, Tuple, Union

def linear_percentage(a: float, b: float, N: int = 10) -> np.ndarray:
    return np.linspace(a, b, N, dtype=float)
    

class BaseDataset_SP(Dataset):
    def __init__(
        self,
        datasets_dir: Path,
        obs_space: DictConfig,
        proprio_state: DictConfig,
        key: str,
        lang_folder: str,
        num_workers: int,
        transforms: Dict = {},
        batch_size: int = 32,
        min_window_size: int = 16,
        max_window_size: int = 32,
        pad: bool = True,
        aux_lang_loss_window: int = 1,
        window_sampling_strategy: str = 'random',
        geometric_p_value: float = 0.1,
        
        load_rgb_depth_vae = False,
        rgb_depth_seq_len = 0,
        rgb_vae_load_path = None,
        depth_vae_load_path = None,
    ):
        ####
        # self.load_rgb_depth_vae = load_rgb_depth_vae
        # self.rgb_depth_seq_len = rgb_depth_seq_len
        # self.rgb_vae_load_path = rgb_vae_load_path
        # self.depth_vae_load_path = depth_vae_load_path
        self.load_rgb_depth_vae = False
        self.rgb_depth_seq_len = False
        self.rgb_vae_load_path = False
        self.depth_vae_load_path = False
        ####


        self.observation_space = obs_space
        self.proprio_state = proprio_state
        self.transforms = transforms
        self.with_lang = key == "lang"
        self.relative_actions = "rel_actions" in self.observation_space["actions"]
        assert window_sampling_strategy in ('random', 'geometric')
        self.window_sampling_strategy = window_sampling_strategy
        self.geometric_p_value = geometric_p_value # only needed for geomtric sampling
        self.pad = pad
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.abs_datasets_dir = Path(datasets_dir)
        self.lang_folder = lang_folder  # if self.with_lang else None
        self.aux_lang_loss_window = aux_lang_loss_window
        assert "validation" in self.abs_datasets_dir.as_posix() or "training" in self.abs_datasets_dir.as_posix()
        self.validation = "validation" in self.abs_datasets_dir.as_posix()
        assert self.abs_datasets_dir.is_dir(), print("self.abs_datasets_dir", self.abs_datasets_dir)
        logger.info(f"loading dataset at {self.abs_datasets_dir}")
        logger.info("finished loading dataset")



        ##########

    def __getitem__(self, idx: Union[int, Tuple[int, int]]) -> Dict:
        """
        Get sequence of dataset.

        Args:
            idx: Index of the sequence.

        Returns:
            Loaded sequence.
        """
        sequence = self._get_sequences(idx)
        return sequence

    def _get_sequences(self, idx: int) -> Dict:
        """
        Load sequence of length window_size.

        Args:
            idx: Index of starting frame.
            window_size: Length of sampled episode.

        Returns:
            dict: Dictionary of tensors of loaded sequence with different input modalities and actions.
        """
        episode = self._load_episode(idx)  # from ExtendedDiskDataset_SP
        # print(list(episode.keys()))

        episode = self.dataset_transform(episode)

        return episode

        seq_state_obs = process_state(episode, self.observation_space, self.transforms, self.proprio_state)
        # print("rgb:", np.min(episode["rgb_static"]), np.max(episode["rgb_static"]))
        seq_rgb_obs = process_rgb(episode, self.observation_space, self.transforms)
        
        # if self.use_depth: 
        #     # print("depth:", np.min(episode["depth_static"]), np.max(episode["depth_static"]))
        #     seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)

        seq_acts = process_actions(episode, self.observation_space, self.transforms)
        info = get_state_info_dict(episode)
        seq_lang = process_language(episode, self.transforms, self.with_lang)
        info = self._add_language_info(info, idx)

        seq_dict = {**seq_state_obs, **seq_rgb_obs, **seq_acts, **info, **seq_lang}  # type:ignore
        seq_dict["idx"] = idx  # type:ignore
        return seq_dict

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def _get_window_size(self, idx: int) -> int:
        """
        Sample a window size taking into account the episode limits.

        Args:
            idx: Index of the sequence to load.

        Returns:
            Window size.
        """
        window_diff = self.max_window_size - self.min_window_size
        if len(self.episode_lookup) <= idx + window_diff:
            # last episode
            max_window = self.min_window_size + len(self.episode_lookup) - idx - 1
        elif self.episode_lookup[idx + window_diff] != self.episode_lookup[idx] + window_diff:
            # less than max_episode steps until next episode
            steps_to_next_episode = int(
                np.nonzero(
                    self.episode_lookup[idx : idx + window_diff + 1]
                    - (self.episode_lookup[idx] + np.arange(window_diff + 1))
                )[0][0]
            )
            max_window = min(self.max_window_size, (self.min_window_size + steps_to_next_episode - 1))
        else:
            max_window = self.max_window_size

        if self.validation:
            # in validation step, repeat the window sizes for each epoch.
            return get_validation_window_size(idx, self.min_window_size, max_window)
        else:
            if self.window_sampling_strategy == 'geometric':
                p = self.geometric_p_value # Choose a suitable value for p
                while True:
                    sampled_window_size = 1 + np.random.geometric(p)
                    if self.min_window_size <= sampled_window_size <= max_window:
                        return sampled_window_size
            else:
                return np.random.randint(self.min_window_size, max_window + 1)

    def __len__(self) -> int:
        """
        Returns:
            Size of the dataset.
        """
        return len(self.episode_lookup)

    def _get_pad_size(self, sequence: Dict) -> int:
        """
        Determine how many frames to append to end of the sequence

        Args:
            sequence: Loaded sequence.

        Returns:
            Number of frames to pad.
        """
        return self.max_window_size - len(sequence["actions"])

    def _pad_sequence(self, seq: Dict, pad_size: int) -> Dict:
        """
        Pad a sequence by repeating the last frame.

        Args:
            seq: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded sequence.
        """
        seq.update({"robot_obs": self._pad_with_repetition(seq["robot_obs"], pad_size)})
        seq.update({"rgb_obs": {k: self._pad_with_repetition(v, pad_size) for k, v in seq["rgb_obs"].items()}})
        seq.update({"depth_obs": {k: self._pad_with_repetition(v, pad_size) for k, v in seq["depth_obs"].items()}})
        #  todo: find better way of distinguishing rk and play action spaces
        if not self.relative_actions:
            # repeat action for world coordinates action space
            seq.update({"actions": self._pad_with_repetition(seq["actions"], pad_size)})
        else:
            # for relative actions zero pad all but the last action dims and repeat last action dim (gripper action)
            seq_acts = torch.cat(
                [
                    self._pad_with_zeros(seq["actions"][..., :-1], pad_size),
                    self._pad_with_repetition(seq["actions"][..., -1:], pad_size),
                ],
                dim=-1,
            )
            seq.update({"actions": seq_acts})
        seq.update({"state_info": {k: self._pad_with_repetition(v, pad_size) for k, v in seq["state_info"].items()}})
        return seq

    @staticmethod
    def _pad_with_repetition(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
        """
        Pad a sequence Tensor by repeating last element pad_size times.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        last_repeated = torch.repeat_interleave(torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0)
        padded = torch.vstack((input_tensor, last_repeated))
        return padded

    @staticmethod
    def _pad_with_zeros(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
        """
        Pad a Tensor with zeros.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0), repeats=pad_size, dim=0
        )
        padded = torch.vstack((input_tensor, zeros_repeated))
        return padded

    def _add_language_info(self, info: Dict, idx: int) -> Dict:
        """
        If dataset contains language, add info to determine if this sequence will be used for the auxiliary losses.

        Args:
            info: Info dictionary.
            idx: Sequence index.

        Returns:
            Info dictionary with updated information.
        """
        if not self.with_lang:
            return info
        use_for_aux_lang_loss = (
            idx + self.aux_lang_loss_window >= len(self.lang_lookup)
            or self.lang_lookup[idx] < self.lang_lookup[idx + self.aux_lang_loss_window]
        )
        info["use_for_aux_lang_loss"] = use_for_aux_lang_loss
        return info



class DiskDataset_SP(BaseDataset_SP):
    """
    Dataset that loads episodes as individual files from disk.

    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        *args: Any,
        skip_frames: int = 1,
        save_format: str = "npz",
        pretrain: bool = False,
        sample_10 = False,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.save_format = save_format
        if self.save_format == "pkl":
            self.load_file = load_pkl
        elif self.save_format == "npz":
            self.load_file = load_npz
        else:
            raise NotImplementedError
        self.pretrain = pretrain
        self.skip_frames = skip_frames

        self.sample_10 = sample_10

        if self.with_lang:  # this
            self.episode_lookup, self.lang_lookup, self.lang_ann, self.lang_text, self.percent_lookup, self.success_percent, self.ep_start_end \
                  = self._build_file_indices_lang(self.abs_datasets_dir)
        else:
            self.episode_lookup = self._build_file_indices(self.abs_datasets_dir)

        self.naming_pattern, self.n_digits = lookup_naming_pattern(self.abs_datasets_dir, self.save_format)




    def _get_episode_name(self, 
                          file_idx,
                          load_type = 'episode') -> Path:
        """
        Convert file idx to file path.

        Args:
            file_idx: index of starting frame.

        Returns:
            Path to file.
        """

        if load_type == 'episode':
            return Path(f"{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}")
        elif load_type == 'rgb_vae':
            return Path(f"{self.rgb_vae_load_path}/episode_{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}")
        elif load_type == 'depth_vae':
            return Path(f"{self.depth_vae_load_path}/episode_{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}")
        
        elif load_type == 'rel_skeleton':
            return PROJECT_ROOT / "calvin" / "backbone" / "backbone_rel" / f"episode_{file_idx:0{self.n_digits}d}.npy"
        
        raise NotImplementedError("")

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.

        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.

        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        start_idx = self.episode_lookup[idx]
        end_idx = start_idx + window_size
        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")
        episodes = [self.load_file(self._get_episode_name(file_idx)) for file_idx in range(start_idx, end_idx)]
        
        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}
        if self.with_lang:
            episode["language"] = self.lang_ann[self.lang_lookup[idx]][0]  # TODO check  [0]
        return episode

    def _build_file_indices_lang(self, abs_datasets_dir: Path) -> Tuple[np.ndarray, List, np.ndarray]:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.

        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.

        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []

        try:
            print("trying to load lang data from: ", abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy")
            lang_data = np.load(abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy", allow_pickle=True).item()
        except Exception:
            print("Exception, trying to load lang data from: ", abs_datasets_dir / "auto_lang_ann.npy")
            lang_data = np.load(abs_datasets_dir / "auto_lang_ann.npy", allow_pickle=True).item()

        ep_start_end_ids = lang_data["info"]["indx"]  # each of them are 64
        lang_ann = lang_data["language"]["emb"]  # length total number of annotations
        lang_text = lang_data["language"]["ann"]  # length total number of annotations

        with open(TAPSAMPLING_ROOT / "complete_percentage.pkl", "rb") as f:
            success_percent = pickle.load(f)



        print("ep_start_end_ids", len(ep_start_end_ids))            # 17870
        print("lang_ann", len(lang_ann))                            # 17870
        print("lang_text", len(lang_text))                          # 17870
        # aaa

        if (not self.validation) and self.sample_10:
            raise NotImplementedError("")
            n = len(ep_start_end_ids)

            import math            

            sample_size = max(1, math.ceil(n * 0.1))
            sample_size = min(sample_size, n)

            file_path = str(PROJECT_ROOT / "calvin_10.json")
            if os.path.exists(file_path):
                print("load form ", file_path)
                with open(file_path, "r", encoding="utf-8") as f:
                    indices = json.load(f)
                indices.sort()
                # print(len(ep_start_end_ids), np.max(indices))
                ep_start_end_ids = [ep_start_end_ids[i] for i in indices]
                lang_ann = [lang_ann[i] for i in indices]
                lang_text = [lang_text[i] for i in indices]

            else:
                print("sample 10% ")
                rng = random.Random(0)
                indices = rng.sample(range(n), k=sample_size)
                indices.sort()
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(indices, f, ensure_ascii=False, indent=2)
                
                ep_start_end_ids = [ep_start_end_ids[i] for i in indices]
                lang_ann = [lang_ann[i] for i in indices]
                lang_text = [lang_text[i] for i in indices]

            print("ep_start_end_ids", len(ep_start_end_ids))            # 1787
            print("lang_ann", len(lang_ann))                            # 1787
            print("lang_text", len(lang_text))                          # 1787
            # aaa

        lang_lookup = []
        percent_lookup = []
        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            # origin_s = start_idx
            # origin_e = end_idx
            
            # if self.pretrain:
            #     start_idx = max(start_idx, end_idx + 1 - self.min_window_size - self.aux_lang_loss_window)
            # assert end_idx >= self.max_window_size
            cnt = 0
            for idx in range(start_idx, end_idx):
                if cnt % self.skip_frames == 0:
                    lang_lookup.append(i)
                    percent_lookup.append( (i, idx-start_idx))      # success_percent中第i个数组，对应的起始位置是 idx-start_idx，后续需要插值获得序列的完成率
                    episode_lookup.append(idx)
                cnt += 1

        return np.array(episode_lookup), lang_lookup, lang_ann, lang_text, percent_lookup, success_percent, ep_start_end_ids

    def _build_file_indices(self, abs_datasets_dir: Path) -> np.ndarray:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the non language
        dataset.

        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.

        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
        """
        raise NotImplementedError("")
        assert abs_datasets_dir.is_dir()

        episode_lookup = []
        ep_start_end_ids = np.load(abs_datasets_dir / "ep_start_end_ids.npy")
        logger.info(f'Found "ep_start_end_ids.npy" with {len(ep_start_end_ids)} episodes.')
        for start_idx, end_idx in ep_start_end_ids:
            assert end_idx > self.max_window_size
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                episode_lookup.append(idx)
        return np.array(episode_lookup)


class ExtendedDiskDataset_SP(DiskDataset_SP):
    def __init__(
        self,
        *args: Any,
        obs_seq_len: int,
        action_seq_len: int,
        future_range: int,
        img_gen_frame_diff: int = 3,

        cond_action_type = ['eepose', 'orient'],     # 'eepose', 'orient', 'skeleton', 三种的组合，但是fullaction包括了eepose


        skeleton_select = [0,1,2,3,4,5,6,7],

        dataset_transform = None,

        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len
        self.future_range = future_range  # Number of steps into the future to sample goals
        self.ep_start_end_ids = np.load(self.abs_datasets_dir / "ep_start_end_ids.npy")  # Load sequence boundaries
        self.img_gen_frame_diff = img_gen_frame_diff
        self.random_frame_diff = False if img_gen_frame_diff > -1 else True 
        self.skeleton_select = skeleton_select

        self.cond_action_type = cond_action_type

        self.dataset_transform = dataset_transform


        # self.min_window_size = self.action_seq_len
        # self.max_window_size = self.action_seq_len + self.future_range
        
    def find_sequence_boundaries(self, idx: int) -> Tuple[int, int]:
        for start_idx, end_idx in self.ep_start_end_ids:
            if start_idx <= idx < end_idx:
                return start_idx, end_idx
        raise ValueError(f"Index {idx} does not belong to any sequence.")

    def _load_episode(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.

        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.

        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """

        start_idx = self.episode_lookup[idx]
        
        pos = self.lang_lookup[idx]
        current_part_start, current_part_end = self.ep_start_end[pos]
        
        # print("idx", idx)
        # print("start_idx", start_idx)
        # print("pos", pos)
        # print("current_part_range", current_part_start, current_part_end)

        # _, part_shift = self.percent_lookup[idx]
        part_sp = self.success_percent[pos]             # (64,)

        target_shift_length = random.randint(3, 12)         #


        part_shift = start_idx - current_part_start
        part_shift_right = current_part_end - start_idx

        if part_shift < target_shift_length:
            end_idx = start_idx + target_shift_length
        elif part_shift_right <= target_shift_length:
            end_idx = start_idx - target_shift_length
        else:
            end_idx = start_idx + random.choice((1, -1)) * target_shift_length

        percent_range = (part_shift, end_idx - current_part_start)
        global_range = (start_idx, end_idx)

        direction = np.sign( global_range[1] - global_range[0] )


        need_interpolate = (target_shift_length != (self.action_seq_len + self.obs_seq_len-1))
        # 如果序列长度和目标长度不同，需要插值

        if need_interpolate:
            ## 从开始到结束，插值完成率
            percent = linear_percentage(part_sp[percent_range[0]], 
                                        part_sp[percent_range[1]],
                                        self.action_seq_len + self.obs_seq_len-1)
        else:
            ## 不需要插值
            percent = part_sp[ percent_range[0] : percent_range[1] : direction ]
            # print("percent_range", percent_range, "percent", percent)


        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.remove("rel_actions")
        keys.append("scene_obs")
        keys.append("actions")
        keys.append("actions")

        
        
        ## 如果direction是正，前面多读一个更小的idx，用于计算相对运动（如果长度不是10，可能需要插值，所以先用绝对运动插值，再计算相对运动）
        ## 如果direction是负，多读一个更大的idx，用于计算相对运动
        episodes = [self.load_file(self._get_episode_name(file_idx)) for file_idx in range(global_range[0], global_range[1], direction)]
        prev_episode = self.load_file(self._get_episode_name(global_range[0] - direction))
        
        # print("loaded:", len(episodes), "range", global_range, "part_range", percent_range)

        episode = {}
        # print("keys:", keys)
        # print("ep_keys", list(episodes[0].keys()))
        for key in keys:
            if 'gen' in key:
                continue
            if 'skeleton' in key:
                continue
            stacked_data = np.stack([ep[key] for ep in episodes])
            if key == "rel_actions" or key == 'actions':
                episode[key] = stacked_data[(self.obs_seq_len-1):, :]
            else:
                episode[key] = stacked_data[0:self.obs_seq_len, :]

        if self.with_lang:
            episode["language"] = self.lang_ann[self.lang_lookup[idx]][0]  
            episode["language_text"] = self.lang_text[self.lang_lookup[idx]] #[0]  

        # for k, v in episode.items():
        #     if isinstance(v, np.ndarray):
        #         print(k, v.shape)
        #     else:
        #         print(k, type(v), v)


        # cond_action_type = ['eepose'],     # 'eepose', 'fullaction', 'gripper', 'skeleton', 三种的组合，但是fullaction包括了eepose

        ### 不需要action插值
        if episode["actions"].shape[0] == (self.action_seq_len + self.obs_seq_len-1):
            rel_actions = []
            for what in self.cond_action_type:
                if 'eepose' == what:
                    absolute_eepose = episode["actions"][:, :3]
                    eepose_prev = np.concatenate([prev_episode["actions"][:3].reshape(1,-1), absolute_eepose[:-1, :]])
                    rel_eepose = (absolute_eepose - eepose_prev) * 50
                    rel_actions.append(rel_eepose)
                if 'orient' == what:
                    absolute_ori = episode["actions"][:, 3:6]
                    ori_prev = np.concatenate([prev_episode["actions"][3:6].reshape(1,-1), absolute_ori[:-1, :]])
                    # print("what", ori_prev, absolute_ori)
                    rel_ori = (absolute_ori - ori_prev) * 20
                    rel_actions.append(rel_ori)
                    # print("rel_ori", rel_ori.shape, rel_ori)
                if 'skeleton' == what:
                    raise NotImplementedError("")
                if 'gripper' == what:
                    raise NotImplementedError("")
                
            rel_actions = np.concatenate(rel_actions, axis = -1)
            ## 10, 6, 没写夹爪
            episode["rel_actions"] = rel_actions
            # why

        else:
            ### 要带着prev一起插值到长度11，再求rel
            rel_actions = []
            prev_current_actions = np.concatenate([prev_episode["actions"].reshape(1,-1), episode["actions"]])  # 11, 7
            for what in self.cond_action_type:
                if 'eepose' == what:
                    interpolated_action_eepose = []
                    for idx in range(0, 3):
                        interpolated_action_eepose.append(interpolate_action(prev_current_actions[:, idx], 
                                                                             target_length = self.action_seq_len + self.obs_seq_len - 1 + 1))
                    interpolated_action_eepose = np.array(interpolated_action_eepose).transpose(1,0)
                    rel_eepose = (interpolated_action_eepose[1:, :3] - interpolated_action_eepose[:-1, :3]) * 50
                    # print("eepose", interpolated_action_eepose.shape, interpolated_action_eepose)
                    # print("rel_eepose", rel_eepose.shape, rel_eepose)
                    # print("prev_episode", prev_episode["rel_actions"])
                    rel_actions.append(rel_eepose)
                    # aaa

                if 'orient' == what:
                    # print("before", prev_current_actions[:, 3:6])
                    interpolated_action_orient = interpolate_orient(prev_current_actions[:, 3:6], 
                                                                    self.action_seq_len + self.obs_seq_len - 1 + 1)
                    
                    # print("after", interpolated_action_orient)
                    rel_orient = (interpolated_action_orient[1:] - interpolated_action_orient[:-1]) * 20
                    # print("orient", interpolated_action_orient.shape, interpolated_action_orient)
                    # print("rel_orient", rel_orient.shape, rel_orient)
                    # print("prev_episode", prev_episode["rel_actions"][3:6])
                    rel_actions.append(rel_orient)
                    # aaaa


            rel_actions = np.concatenate(rel_actions, -1)
            episode["rel_actions"] = rel_actions

        episode['success_percent'] = percent

        # for k, v in episode.items():
        #     if isinstance(v, np.ndarray):
        #         print(k, v.shape, v)
        #     else:
        #         print(k, v)

        return episode
    
       
    
    def merge_episodes(self, episode1: Dict[str, np.ndarray], episode2: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        merged_episode = {}
        all_keys = set(episode1.keys()).union(set(episode2.keys()))
        for key in all_keys:
            if key in episode1 and key in episode2:
                # Merge logic here, for example:
                merged_episode[key] = np.concatenate([episode1[key], episode2[key]], axis=0)
            elif key in episode1:
                merged_episode[key] = episode1[key]
            else:
                merged_episode[key] = episode2[key]
        return merged_episode
    
    def _build_file_indices(self, abs_datasets_dir: Path) -> np.ndarray:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the non language
        dataset.

        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.

        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []

        ep_start_end_ids = np.load(abs_datasets_dir / "ep_start_end_ids.npy")
        logger.info(f'Found "ep_start_end_ids.npy" with {len(ep_start_end_ids)} episodes.')
        for start_idx, end_idx in ep_start_end_ids:
            assert end_idx > self.max_window_size
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                episode_lookup.append(idx)
        return np.array(episode_lookup)
