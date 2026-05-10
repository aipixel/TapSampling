import sys
import torch
import numpy as np
from pathlib import Path
from copy import deepcopy
from typing import Any, Dict, Tuple, Union
from policy_models.datasets.utils.episode_utils import process_depth, process_rgb, process_state
import torch.nn.functional as F
import requests
import time

class GetActionsFromPort():
    def __init__(self):
        return
    
    def sample(self, actions, sample_num):
        
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy() 

        payload = {
            "actions": actions.tolist(), 
            "M": sample_num,
        }

        for attempt in range(1, 10):
            try:
                resp = requests.post("http://127.0.0.1:8000/generate", json=payload, timeout=60)
                new_actions = np.array(resp.json()['generated'])
                return new_actions
            except requests.exceptions.ConnectionError as e:
                print(f"Connection failed (attempt {attempt}/5): {e}")
            except requests.exceptions.Timeout as e:
                print(f"Request timed out (attempt {attempt}/5): {e}")
            except requests.exceptions.HTTPError as e:
                print(f"HTTP error: {e}")
                break
            time.sleep(min(2**attempt, 30))
        else:
            raise RuntimeError("Unable to connect to the service after multiple attempts")

# For VPP
class SamplingFramework:
    def __init__(   self, 
                    policy, 
                    verifier,
                    same_transform,
                    policy_transform,
                    verifier_transform,
                    sample_config = {
                                'sample_type': 'policy',
                                'policy_sample_num': 4,
                                'total_sample_num': 16,
                                'max_batch_num': 8
                            },
                    selection_config = {
                                'selection_type': 'greedy',
                                'selection_direction': 'good',
                            },
                    device = None,
                    proprio_state = None,
                    observation_space = None
                    ):
        self.policy = policy
        self.verifier = verifier

        self.sample_config = sample_config
        self.selection_config = selection_config

        self.same_transform = same_transform
        self.policy_transform = policy_transform
        self.verifier_transform = verifier_transform

        self.device = device
        self.proprio_state = proprio_state
        self.observation_space_keys = observation_space
        
        self.action_vae = GetActionsFromPort()


    def transform_observation(self, obs, transforms) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        obs['rgb_obs']['cond_static'] = obs['rgb_obs']['rgb_static']
        obs['rgb_obs']['cond_gripper'] = obs['rgb_obs']['rgb_gripper']
        state_obs = process_state(obs, self.observation_space_keys, transforms, self.proprio_state)
        rgb_obs = process_rgb(obs["rgb_obs"], self.observation_space_keys, transforms)
        depth_obs = process_depth(obs["depth_obs"], self.observation_space_keys, transforms)

        state_obs["robot_obs"] = state_obs["robot_obs"].to(self.device).unsqueeze(0)
        rgb_obs.update({"rgb_obs": {k: v.to(self.device).unsqueeze(0) for k, v in rgb_obs["rgb_obs"].items()}})
        depth_obs.update({"depth_obs": {k: v.to(self.device).unsqueeze(0) for k, v in depth_obs["depth_obs"].items()}})

        obs_dict: Dict = {
            **rgb_obs,
            **state_obs,
            **depth_obs,
            # "robot_obs_raw": torch.from_numpy(obs["robot_obs"]).to(self.device),
        }
        return obs_dict

    def step(self, obs, goal):
        
        obs_p = self.transform_observation(obs, self.policy_transform)

        obs_sp_static =  obs_p["rgb_obs"]["rgb_static"].detach().cpu().squeeze(0) # 1,3,256,256
        obs_sp_static = F.interpolate(obs_sp_static, size=(200, 200), mode='bilinear', align_corners=False)
        obs_sp_static = (obs_sp_static.squeeze(0).permute(1,2,0).numpy() * 128 + 128).clip(0, 255).astype(np.uint8)
        obs_sp_gripper =  obs_p["rgb_obs"]["rgb_gripper"].detach().cpu().squeeze(0) # 1,3,256,256
        obs_sp_gripper = F.interpolate(obs_sp_gripper, size=(84, 84), mode='bilinear', align_corners=False)
        obs_sp_gripper = (obs_sp_gripper.squeeze(0).permute(1,2,0).numpy() * 128 + 128).clip(0, 255).astype(np.uint8)

        obs_v = dict(
            rgb_obs=dict(rgb_static = obs_sp_static, rgb_gripper = obs_sp_gripper)
        )

        all_actions = []
        all_scores = []

        if self.sample_config['sample_type'] == 'policy':
            for forward_idx in range(0, self.sample_config['total_sample_num'], self.sample_config['max_batch_num']):
                forward_batch_num = min(self.sample_config['max_batch_num'], 
                                    self.sample_config['total_sample_num'] - forward_idx)
                batch_action_chunks = self.policy.step(obs_p, 
                                                        goal,
                                                        return_chunk = True,
                                                        sample_num = forward_batch_num)       # b, 10, 7
                batch_score = self.verifier.step(obs_v, 
                                                goal['lang_text'], 
                                                0, 
                                                query_actions = batch_action_chunks.to(torch.bfloat16).to(torch.device("cuda:0"))  )
                
                all_actions.append(batch_action_chunks.detach().cpu().numpy())
                all_scores.append(batch_score.squeeze(-1).mean(-1))

            all_actions = np.concatenate(all_actions, axis = 0)
            all_scores = np.concatenate(all_scores, axis = 0)

        elif self.sample_config['sample_type'] == 'vae':
            for forward_idx in range(0, self.sample_config['policy_sample_num'], self.sample_config['max_batch_num']):
                forward_batch_num = min(self.sample_config['max_batch_num'], 
                                    self.sample_config['policy_sample_num'] - forward_idx)
                batch_action_chunks = self.policy.step(obs_p, 
                                                        goal,
                                                        return_chunk = True,
                                                        sample_num = forward_batch_num)       # b, 10, 7
                all_actions.append(batch_action_chunks.detach().cpu().numpy())
            all_actions = np.concatenate(all_actions, axis = 0)

            if self.sample_config['total_sample_num'] - self.sample_config['policy_sample_num'] > 0:
                new_actions = self.action_vae.sample(all_actions, 
                                                    self.sample_config['total_sample_num'] - self.sample_config['policy_sample_num'])
                all_actions = np.concatenate([all_actions, new_actions], axis = 0)

            batch_score = self.verifier.step(obs_v, 
                                            goal['lang_text'], 
                                            0, 
                                            query_actions = torch.from_numpy(all_actions).to(torch.bfloat16).to(torch.device("cuda:0"))  )
            
            all_scores.append(batch_score.squeeze(-1).mean(-1))
            all_scores = np.concatenate(all_scores, axis = 0)

        else:
            print("current sample_config:", self.sample_config)
            raise NotImplementedError("")

        return self.select_actions(all_actions, all_scores)
    
    def select_actions(self, all_actions, all_scores):

        # all_actions = np.concatenate(all_actions, axis = 0)     # n, 10, 7
        # all_scores = np.concatenate(all_scores, axis = 0)       # n, 

        # print("all_actions", all_actions.shape)
        # print("all_scores", all_scores.shape)
        # print("all_actions", all_actions)
        # print("all_scores", all_scores)
        # aaa

        if self.selection_config['selection_type'] == 'greedy':
            if self.selection_config['selection_direction'] == 'good':
                idxs = np.argsort(all_scores)[-1:]
            else:
                idxs = np.argsort(all_scores)[:1]
            return all_scores[idxs][0], torch.from_numpy(all_actions[idxs].squeeze(0))
        

        elif self.selection_config['selection_type'] == 'thres_weighted_merge':
            return self.select_or_weighted_average(all_actions, all_scores, threshold = self.selection_config['selection_threshold'])

        elif self.selection_config['selection_type'] == 'thres_minimal_3':
            return self.thres_minimal_3(all_actions, all_scores, threshold = self.selection_config['selection_threshold'])
        
        else:
            print("current selection_config:", self.selection_config)
            raise NotImplementedError("")

    def reset(self):
        self.policy.reset()

    def select_or_weighted_average(self, actions: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
        """
        Args:
        actions: np.ndarray, shape (n, 10, 7)
        scores:  np.ndarray, shape (n,)
        threshold: float

        Returns:
        A np.ndarray with shape (10, 7):
            - If no score is >= threshold, return the action at argmax(score).
            - Otherwise, return the score-weighted average of all actions with score >= threshold.
        """
        actions = np.asarray(actions)
        scores = np.asarray(scores)

        if actions.ndim != 3 or actions.shape[0] != scores.shape[0]:
            raise ValueError("actions must be (n,10,7) and scores must be (n,) with matching n")

        # Find indices that satisfy the threshold.
        idxs = np.where(scores >= threshold)[0]

        if idxs.size == 0:
            best_idx = int(np.argmax(scores))
            # print("all lower than thres", scores)
            return np.max(scores), torch.from_numpy(actions[best_idx])  # shape (10,7)

        # If the threshold is met, compute a score-weighted average.
        weights = scores[idxs].astype(float)
        weight_sum = weights.sum()

        if weight_sum == 0:
            # Avoid division by zero by falling back to a uniform average.
            # print("noway", weights)
            return 0.0, torch.from_numpy(actions[idxs].mean(axis=0))

        # Compute the weighted sum efficiently with tensordot.
        weighted_sum = np.tensordot(weights, actions[idxs], axes=(0, 0))  # shape (10,7)

        # print("weighted mean", np.mean(weights))

        return np.mean(weights), torch.from_numpy(weighted_sum / weight_sum)

    def thres_minimal_3(self, actions: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    
        actions = np.asarray(actions)
        scores = np.asarray(scores)

        if actions.ndim != 3 or actions.shape[0] != scores.shape[0]:
            raise ValueError("actions must be (n,10,7) and scores must be (n,) with matching n")

        idxs = np.where(scores >= threshold)[0]

        if idxs.size == 0:
            # No action reaches the threshold: return the highest-scoring action.
            best_idx = int(np.argmax(scores))
            return float(np.max(scores)), torch.from_numpy(actions[best_idx].astype(np.float32))

        # Among threshold-qualified candidates, select the three lowest scores.
        # Sort scores in ascending order within the idxs subset.
        sub_scores = scores[idxs]
        order = np.argsort(sub_scores)  # ascending indices
        # Take the first three, or all of them if fewer than three exist.
        take_k = min(3, order.size)
        chosen_local_idxs = order[:take_k]     # local indices within sub_scores / idxs
        chosen_idxs = idxs[chosen_local_idxs]  # convert back to global indices

        selected_actions = actions[chosen_idxs]  # shape (k, 10, 7)
        avg_action = selected_actions.mean(axis=0)  # simple unweighted average
        avg_score = float(sub_scores[chosen_local_idxs].mean())

        return avg_score, torch.from_numpy(avg_action.astype(np.float32))

    def step_only_policy(self, obs, goal):
        """
        Run the policy only, without verification or selection, for speed testing.
        """
        obs_p = self.transform_observation(obs, self.policy_transform)

        obs_sp_static =  obs_p["rgb_obs"]["rgb_static"].detach().cpu().squeeze(0) # 1,3,256,256
        obs_sp_static = F.interpolate(obs_sp_static, size=(200, 200), mode='bilinear', align_corners=False)
        obs_sp_static = (obs_sp_static.squeeze(0).permute(1,2,0).numpy() * 128 + 128).clip(0, 255).astype(np.uint8)
        obs_sp_gripper =  obs_p["rgb_obs"]["rgb_gripper"].detach().cpu().squeeze(0) # 1,3,256,256
        obs_sp_gripper = F.interpolate(obs_sp_gripper, size=(84, 84), mode='bilinear', align_corners=False)
        obs_sp_gripper = (obs_sp_gripper.squeeze(0).permute(1,2,0).numpy() * 128 + 128).clip(0, 255).astype(np.uint8)

        all_actions = []

        if self.sample_config['sample_type'] == 'policy':
            for forward_idx in range(0, self.sample_config['total_sample_num'], self.sample_config['max_batch_num']):
                
                forward_batch_num = min(self.sample_config['max_batch_num'], 
                                    self.sample_config['total_sample_num'] - forward_idx)
                batch_action_chunks = self.policy.step(obs_p, 
                                                        goal,
                                                        return_chunk = True,
                                                        sample_num = forward_batch_num)       # b, 10, 7
                all_actions.append(batch_action_chunks.detach().cpu().numpy())
            all_actions = np.concatenate(all_actions, axis = 0)


        elif self.sample_config['sample_type'] == 'vae':
            batch_action_chunks_policy = self.policy.step(obs_p,
                                                    goal,
                                                    return_chunk = True,
                                                    sample_num = self.sample_config['policy_sample_num'])
            policy_actions = batch_action_chunks_policy.detach().cpu().numpy()
            all_actions.append(policy_actions)

            extra_num = self.sample_config['total_sample_num'] - self.sample_config['policy_sample_num']
            if extra_num > 0:
                batch_action_chunks_sample = self.action_vae.sample(policy_actions, extra_num)
                all_actions.append(batch_action_chunks_sample)

            all_actions = np.concatenate(all_actions, axis=0)

        else:
            print("current sample_config:", self.sample_config)
            raise NotImplementedError("")

        return all_actions
    
    def step_only_verifier(self, obs, goal, all_actions):
        """
        Used to test verifier speed.
        """

        obs_p = self.transform_observation(obs, self.policy_transform)

        obs_sp_static =  obs_p["rgb_obs"]["rgb_static"].detach().cpu().squeeze(0) # 1,3,256,256
        obs_sp_static = F.interpolate(obs_sp_static, size=(200, 200), mode='bilinear', align_corners=False)
        obs_sp_static = (obs_sp_static.squeeze(0).permute(1,2,0).numpy() * 128 + 128).clip(0, 255).astype(np.uint8)
        obs_sp_gripper =  obs_p["rgb_obs"]["rgb_gripper"].detach().cpu().squeeze(0) # 1,3,256,256
        obs_sp_gripper = F.interpolate(obs_sp_gripper, size=(84, 84), mode='bilinear', align_corners=False)
        obs_sp_gripper = (obs_sp_gripper.squeeze(0).permute(1,2,0).numpy() * 128 + 128).clip(0, 255).astype(np.uint8)

        obs_v = dict(
            rgb_obs=dict(rgb_static = obs_sp_static, rgb_gripper = obs_sp_gripper)
        )

        batch_score = self.verifier.step(obs_v, 
                                        goal['lang_text'], 
                                        0, 
                                        query_actions = torch.from_numpy(all_actions).to(torch.bfloat16).to(torch.device("cuda:0"))  )


# For DiffusionPolicy
class SamplingFramework_DP:
    def __init__(   self, 
                    
                    policy, 
                    verifier,

                    same_transform,
                    policy_transform,
                    verifier_transform,
                    
                    sample_config = {
                                'sample_type': 'policy',
                                'policy_sample_num': 4,
                                'total_sample_num': 16,
                                'max_batch_num': 8
                            },
                    selection_config = {
                                'selection_type': 'greedy',
                                'selection_direction': 'good',   # good or bad
                            },

                    device = None,
                    proprio_state = None,
                    observation_space = None
                    ):
        self.policy = policy
        self.verifier = verifier

        self.sample_config = sample_config
        self.selection_config = selection_config

        self.policy_transform = policy_transform
        self.verifier_transform = verifier_transform

        self.device = device
        self.proprio_state = proprio_state
        self.observation_space_keys = observation_space
        self.action_vae = GetActionsFromPort()
        


    def transform_observation(self, obs, transforms) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        obs['rgb_obs']['cond_static'] = obs['rgb_obs']['rgb_static']
        obs['rgb_obs']['cond_gripper'] = obs['rgb_obs']['rgb_gripper']
        state_obs = process_state(obs, self.observation_space_keys, transforms, self.proprio_state)
        rgb_obs = process_rgb(obs["rgb_obs"], self.observation_space_keys, transforms)
        depth_obs = process_depth(obs["depth_obs"], self.observation_space_keys, transforms)

        state_obs["robot_obs"] = state_obs["robot_obs"].to(self.device).unsqueeze(0)
        rgb_obs.update({"rgb_obs": {k: v.to(self.device).unsqueeze(0) for k, v in rgb_obs["rgb_obs"].items()}})
        depth_obs.update({"depth_obs": {k: v.to(self.device).unsqueeze(0) for k, v in depth_obs["depth_obs"].items()}})

        obs_dict: Dict = {
            **rgb_obs,
            **state_obs,
            **depth_obs,
            # "robot_obs_raw": torch.from_numpy(obs["robot_obs"]).to(self.device),
        }
        return obs_dict

    def step(self, obs, goal):
        
        obs_p = self.transform_observation(obs, self.verifier_transform)
        obs_sp_static =  obs_p["rgb_obs"]["rgb_static"].detach().cpu().squeeze(0) # 1,3,256,256
        obs_sp_static = F.interpolate(obs_sp_static, size=(200, 200), mode='bilinear', align_corners=False)
        obs_sp_static = (obs_sp_static.squeeze(0).permute(1,2,0).numpy() * 128 + 128).clip(0, 255).astype(np.uint8)
        obs_sp_gripper =  obs_p["rgb_obs"]["rgb_gripper"].detach().cpu().squeeze(0) # 1,3,256,256
        obs_sp_gripper = F.interpolate(obs_sp_gripper, size=(84, 84), mode='bilinear', align_corners=False)
        obs_sp_gripper = (obs_sp_gripper.squeeze(0).permute(1,2,0).numpy() * 128 + 128).clip(0, 255).astype(np.uint8)
        obs_v = dict(
            rgb_obs=dict(rgb_static = obs_sp_static, rgb_gripper = obs_sp_gripper)
        )

        obs_p = self.transform_observation(obs, self.policy_transform)
        obs_p["rgb_obs"]["rgb_static"] = obs_p["rgb_obs"]["rgb_static"].squeeze(0)
        obs_p["rgb_obs"]["rgb_gripper"] = obs_p["rgb_obs"]["rgb_gripper"].squeeze(0)

        # print("obs_pppppppppp")
        # print("obs_p", obs_p["rgb_obs"]["rgb_static"].shape, obs_p["rgb_obs"]["rgb_gripper"].shape)
        # print("obs_vvvvvvvvvv")
        # print("obs_v", obs_v["rgb_obs"]["rgb_static"].shape, obs_v["rgb_obs"]["rgb_gripper"].shape)
        # aaa

        if self.sample_config['total_sample_num'] == 1:
            action = self.policy.step(obs_p, 
                                    goal,
                                    return_chunk = True,
                                    sample_num = 1)
            
            return 0.0, action

        all_actions = []
        all_scores = []

        if self.sample_config['sample_type'] == 'policy':
            policy_forward_time = (self.sample_config['total_sample_num'] // self.sample_config['max_batch_num'])
            if self.sample_config['total_sample_num'] % self.sample_config['max_batch_num'] != 0:
                policy_forward_time += 1
            
            for forward_idx in range(0, self.sample_config['total_sample_num'], self.sample_config['max_batch_num']):
                
                forward_batch_num = min(self.sample_config['max_batch_num'], 
                                    self.sample_config['total_sample_num'] - forward_idx)
                
                batch_action_chunks = self.policy.step(obs_p, 
                                                        goal,
                                                        return_chunk = True,
                                                        sample_num = forward_batch_num)       # b, 10, 7

                batch_score = self.verifier.step(obs_v, 
                                                goal['lang_text'], 
                                                0, 
                                                query_actions = batch_action_chunks.to(torch.bfloat16).to(torch.device("cuda:0"))  )
                
                all_actions.append(batch_action_chunks.detach().cpu().numpy())
                all_scores.append(batch_score.squeeze(-1).mean(-1))

            all_actions = np.concatenate(all_actions, axis = 0)
            all_scores = np.concatenate(all_scores, axis = 0)

        elif self.sample_config['sample_type'] == 'vae':
            batch_action_chunks_policy = self.policy.step(obs_p,
                                                    goal,
                                                    return_chunk = True,
                                                    sample_num = self.sample_config['policy_sample_num'])
            policy_actions = batch_action_chunks_policy.detach().cpu().numpy()
            all_actions.append(policy_actions)

            extra_num = self.sample_config['total_sample_num'] - self.sample_config['policy_sample_num']
            if extra_num > 0:
                batch_action_chunks_sample = self.action_vae.sample(policy_actions, extra_num)
                all_actions.append(batch_action_chunks_sample)

            all_actions = np.concatenate(all_actions, axis=0)

            batch_score = self.verifier.step(obs_v,
                                            goal['lang_text'],
                                            0,
                                            query_actions = torch.from_numpy(all_actions).to(torch.bfloat16).to(torch.device("cuda:0")))

            all_scores.append(batch_score.squeeze(-1).mean(-1))
            all_scores = np.concatenate(all_scores, axis=0)

        else:
            print("current sample_config:", self.sample_config)
            raise NotImplementedError("")
        
        # print("all_actions", all_actions.shape)
        # print("all_scores", all_scores.shape)
        # aaa

        return self.select_actions(all_actions, all_scores)
    
    def select_actions(self, all_actions, all_scores):

        # all_actions = np.concatenate(all_actions, axis = 0)     # n, 10, 7
        # all_scores = np.concatenate(all_scores, axis = 0)       # n, 

        # print("all_actions", all_actions.shape)
        # print("all_scores", all_scores.shape)
        # print("all_actions", all_actions)
        # print("all_scores", all_scores)
        # aaa

        if self.selection_config['selection_type'] == 'greedy':
            if self.selection_config['selection_direction'] == 'good':
                idxs = np.argsort(all_scores)[-1:]
            else:
                idxs = np.argsort(all_scores)[:1]
            return all_scores[idxs][0], torch.from_numpy(all_actions[idxs].squeeze(0))
        

        elif self.selection_config['selection_type'] == 'thres_weighted_merge':
            return self.select_or_weighted_average(all_actions, all_scores, threshold = self.selection_config['selection_threshold'])

        elif self.selection_config['selection_type'] == 'thres_minimal_3':
            return self.thres_minimal_3(all_actions, all_scores, threshold = self.selection_config['selection_threshold'])
        
        else:
            print("current selection_config:", self.selection_config)
            raise NotImplementedError("")

    def reset(self):
        self.policy.reset()


    def select_or_weighted_average(self, actions: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
        """
        Args:
        actions: np.ndarray, shape (n, 10, 7)
        scores:  np.ndarray, shape (n,)
        threshold: float

        Returns:
        A np.ndarray with shape (10, 7):
            - If no score is >= threshold, return the action at argmax(score).
            - Otherwise, return the score-weighted average of all actions with score >= threshold.
        """
        actions = np.asarray(actions)
        scores = np.asarray(scores)

        if actions.ndim != 3 or actions.shape[0] != scores.shape[0]:
            raise ValueError("actions must be (n,10,7) and scores must be (n,) with matching n")

        # Find indices that satisfy the threshold.
        idxs = np.where(scores >= threshold)[0]

        if idxs.size == 0:
            best_idx = int(np.argmax(scores))
            # print("all lower than thres", scores)
            return np.max(scores), torch.from_numpy(actions[best_idx])  # shape (10,7)

        # If the threshold is met, compute a score-weighted average.
        weights = scores[idxs].astype(float)
        weight_sum = weights.sum()

        if weight_sum == 0:
            # Avoid division by zero by falling back to a uniform average.
            # print("noway", weights)
            return 0.0, torch.from_numpy(actions[idxs].mean(axis=0))

        # Compute the weighted sum efficiently with tensordot.
        weighted_sum = np.tensordot(weights, actions[idxs], axes=(0, 0))  # shape (10,7)

        # print("weighted mean", np.mean(weights))

        return np.mean(weights), torch.from_numpy(weighted_sum / weight_sum)

    def thres_minimal_3(self, actions: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    
        actions = np.asarray(actions)
        scores = np.asarray(scores)

        if actions.ndim != 3 or actions.shape[0] != scores.shape[0]:
            raise ValueError("actions must be (n,10,7) and scores must be (n,) with matching n")

        idxs = np.where(scores >= threshold)[0]

        if idxs.size == 0:
            # No action reaches the threshold: return the highest-scoring action.
            best_idx = int(np.argmax(scores))
            return float(np.max(scores)), torch.from_numpy(actions[best_idx].astype(np.float32))

        # Among threshold-qualified candidates, select the three lowest scores.
        # Sort scores in ascending order within the idxs subset.
        sub_scores = scores[idxs]
        order = np.argsort(sub_scores)  # ascending indices
        # Take the first three, or all of them if fewer than three exist.
        take_k = min(3, order.size)
        chosen_local_idxs = order[:take_k]     # local indices within sub_scores / idxs
        chosen_idxs = idxs[chosen_local_idxs]  # convert back to global indices

        selected_actions = actions[chosen_idxs]  # shape (k, 10, 7)
        avg_action = selected_actions.mean(axis=0)  # simple unweighted average
        avg_score = float(sub_scores[chosen_local_idxs].mean())

        return avg_score, torch.from_numpy(avg_action.astype(np.float32))
