import os.path
from itertools import chain
import logging
from pathlib import Path
import pickle
from typing import Any, Dict, List, Tuple
import random

from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
import numpy as np

from policy_models.datasets.base_dataset import BaseDataset
from policy_models.datasets.utils.episode_utils import lookup_naming_pattern
import json
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_pkl(filename: Path) -> Dict[str, np.ndarray]:
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_npz(filename: Path) -> Dict[str, np.ndarray]:
    return np.load(filename.as_posix())


class DiskDataset(BaseDataset):
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
            self.episode_lookup, self.lang_lookup, self.lang_ann, self.lang_text = self._build_file_indices_lang(self.abs_datasets_dir)
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

        print("ep_start_end_ids", len(ep_start_end_ids))            # 17870
        print("lang_ann", len(lang_ann))                            # 17870
        print("lang_text", len(lang_text))                          # 17870
        # aaa

        if (not self.validation) and self.sample_10:
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
        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            if self.pretrain:
                start_idx = max(start_idx, end_idx + 1 - self.min_window_size - self.aux_lang_loss_window)
            assert end_idx >= self.max_window_size
            cnt = 0
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                if cnt % self.skip_frames == 0:
                    lang_lookup.append(i)
                    episode_lookup.append(idx)
                cnt += 1

        return np.array(episode_lookup), lang_lookup, lang_ann, lang_text

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


class ExtendedDiskDataset(DiskDataset):
    def __init__(
        self,
        *args: Any,
        obs_seq_len: int,
        action_seq_len: int,
        future_range: int,
        img_gen_frame_diff: int = 3,


        skeleton_select = [0,1,2,3,4,5,6,7],

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
        # self.min_window_size = self.action_seq_len
        # self.max_window_size = self.action_seq_len + self.future_range
        
    def find_sequence_boundaries(self, idx: int) -> Tuple[int, int]:
        for start_idx, end_idx in self.ep_start_end_ids:
            if start_idx <= idx < end_idx:
                return start_idx, end_idx
        raise ValueError(f"Index {idx} does not belong to any sequence.")

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
        end_idx = start_idx + self.action_seq_len + self.obs_seq_len-1
        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")
        episodes = [self.load_file(self._get_episode_name(file_idx)) for file_idx in range(start_idx, end_idx)]
        # ['actions', 'rel_actions', 'robot_obs', 'scene_obs', 'rgb_static', 'rgb_gripper', 'rgb_tactile', 'depth_static', 'depth_gripper', 'depth_tactile']
        # print(list(episodes[0].keys()))
        # aa
        episode = {}
        # print("keys:", keys)
        for key in keys:
            if 'gen' in key:
                continue
            
            if 'skeleton' in key:
                continue

            stacked_data = np.stack([ep[key] for ep in episodes])
            if key == "rel_actions" or key == 'actions':
                episode[key] = stacked_data[(self.obs_seq_len-1):((self.obs_seq_len-1) + self.action_seq_len), :]
            elif (key == 'depth_static'):
                episode[key] = stacked_data[(self.obs_seq_len-1):((self.obs_seq_len-1) + self.depth_seq_len), :]
            else:
                episode[key] = stacked_data[:self.obs_seq_len, :]

        if self.with_lang:
            episode["language"] = self.lang_ann[self.lang_lookup[idx]][0]  # TODO check  [0]
            episode["language_text"] = self.lang_text[self.lang_lookup[idx]] #[0]  # TODO check  [0]
        
        # get the random future state as goal
        # goal_idx = end_idx + window_size
        # # print(start_idx, end_idx, goal_idx)
        # eps_start_idx, eps_end_idx = self.find_sequence_boundaries(end_idx)
        #
        # # Check if future goal can be sampled
        #
        # if eps_end_idx < goal_idx:
        #     goal_idx = eps_end_idx
        
        # goal_episodes = self.load_file(self._get_episode_name(goal_idx))
        # goal_episode = {}
        # for key in keys:
        #     if 'gen' in key:
        #         continue
        #     goal_stacked_data = np.stack([goal_episodes[key]])
        #     if key == "rel_actions" or key == 'actions':
        #         pass
        #     else:
        #         goal_episode[key] = goal_stacked_data[:self.obs_seq_len, :]
        # # store for merging
        #
        # episode = self.merge_episodes(episode, goal_episode)


        # action对应的是接下来的移动pose，此处读取的rel_skeleton是上一帧的相对移动
        # 所以读取 idx+1 的 rel_skeleton，代表着未来N帧中骨骼的移动
        if "rel_skeleton" in keys:
            skeletons_episodes = [np.load(self._get_episode_name(file_idx, 'rel_skeleton')) for file_idx in range(start_idx+1, end_idx+1)]
            rel_skeleton = np.concatenate(skeletons_episodes, axis = 0)
            rel_skeleton = rel_skeleton[(self.obs_seq_len-1):((self.obs_seq_len-1) + self.action_seq_len):, self.skeleton_select, :]
            rel_skeleton = rel_skeleton.reshape(rel_skeleton.shape[0], -1)
            # print("start_idx", start_idx, "rel_skeleton:", rel_skeleton.shape)
            # print("start_idx", start_idx, "rel_actions:", episode["rel_actions"].shape)
            # aaa
            episode["rel_actions"] = np.concatenate([episode["rel_actions"], rel_skeleton], axis = -1)
            # print("after_concat:", episode["rel_actions"].shape)
            # aa
            #####
        if self.load_rgb_depth_vae:
            start_idx = self.episode_lookup[idx]
            end_idx = start_idx + self.rgb_depth_seq_len    #self.action_seq_len + self.obs_seq_len-1
            # print([self._get_episode_name(file_idx, 'rgb_vae') for file_idx in range(start_idx, end_idx)])
            # print([self._get_episode_name(file_idx, 'depth_vae') for file_idx in range(start_idx, end_idx)])
            
            # little mistake in saving npz...
            #rgb_vae_episodes = [load_npz(self._get_episode_name(file_idx, 'rgb_vae'))["depth_static"] for file_idx in range(start_idx, end_idx)]
            rgb_vae_episodes = [load_npz(self._get_episode_name(start_idx, 'rgb_vae'))["depth_static"]] 
            # b × (1,1,4,32,32), 
            depth_vae_episodes = [load_npz(self._get_episode_name(file_idx, 'depth_vae'))["depth_static"] for file_idx in range(start_idx, end_idx)]
            # b × (1,t,4,32,32), 

            episode["rgb_vae"] = np.stack(rgb_vae_episodes).squeeze(1)      #(b,1,4,32,32)
            episode["depth_vae"] = np.stack(depth_vae_episodes).squeeze(1)  #(b,t,4,32,32)

            # print("rgb_vae:", episode["rgb_vae"].shape, episode["rgb_vae"][0,0,0])
            # print("depth_vae:", episode["depth_vae"].shape, episode["depth_vae"][0,0,0])
            # aaa

        # for k, v in episode.items():
        #     if isinstance(v, np.ndarray):
        #         print(k, v.shape, v)
        #     else:
        #         print(k, v)


        # aaa
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
