from collections import Counter, defaultdict
import json
import logging
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TAPSAMPLING_ROOT = PROJECT_ROOT.parent / "tapsampling"
POLICY_CONF_DIR = PROJECT_ROOT / "policy_conf"
DEFAULT_VERIFIER_CHECKPOINT = (
    TAPSAMPLING_ROOT
    / "outputs_actiondim7"
    / "configs+calvin_abc+b16+lr-1e-05+lora-r64+dropout-0.0--image_aug--VLA-Adapter--calvin--actiondim7--65000_chkpt"
)

# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, PROJECT_ROOT.as_posix())

import hydra
import numpy as np
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm
import wandb
import torch.distributed as dist

from policy_evaluation.multistep_sequences import get_sequences
from policy_evaluation.utils import get_default_beso_and_env, get_env_state_for_initial_condition, join_vis_lang
from policy_models.utils.utils import get_last_checkpoint, get_all_checkpoints
from policy_models.rollout.rollout_video import RolloutVideo
from policy_evaluation.tapsampling_imports import DualSystemCalvinEvaluation_SP







logger = logging.getLogger(__name__)
# wandb.init()

def get_video_tag(i):
    if dist.is_available() and dist.is_initialized():
        i = i * dist.get_world_size() + dist.get_rank()
    return f"_long_horizon/sequence_{i}"


def get_log_dir(log_dir):
    if log_dir is not None:
        log_dir = Path(log_dir)
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = Path(__file__).parents[3] / "evaluation"
        if not log_dir.exists():
            log_dir = Path("/tmp/evaluation")

    log_dir = log_dir / "logs" / time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(log_dir, exist_ok=False)
    print(f"logging to {log_dir}")
    return log_dir


def count_success(results):
    count = Counter(results)
    step_success = []
    for i in range(1, 6):
        n_success = sum(count[j] for j in reversed(range(i, 6)))
        sr = n_success / len(results)
        step_success.append(sr)
    return step_success


def print_and_save(total_results, plan_dicts, cfg, log_dir=None):
    if log_dir is None:
        log_dir = get_log_dir(cfg.eval_save_dir)

    sequences = get_sequences(cfg.num_sequences)

    current_data = {}
    ranking = {}
    for checkpoint, results in total_results.items():
        checkpoint = str(checkpoint)
        if "=" in checkpoint:
            epoch = checkpoint.stem.split("=")[1]
        else:
            epoch = checkpoint
        print(f"Results for Epoch {epoch}:")
        avg_seq_len = np.mean(results)
        ranking[epoch] = avg_seq_len
        chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results))}
        print(f"Average successful sequence length: {avg_seq_len}")
        print("Success rates for i instructions in a row:")
        for i, sr in chain_sr.items():
            print(f"{i}: {sr * 100:.1f}%")

        cnt_success = Counter()
        cnt_fail = Counter()

        for result, (_, sequence) in zip(results, sequences):
            for successful_tasks in sequence[:result]:
                cnt_success[successful_tasks] += 1
            if result < len(sequence):
                failed_task = sequence[result]
                cnt_fail[failed_task] += 1

        total = cnt_success + cnt_fail
        task_info = {}
        for task in total:
            task_info[task] = {"success": cnt_success[task], "total": total[task]}
            print(f"{task}: {cnt_success[task]} / {total[task]} |  SR: {cnt_success[task] / total[task] * 100:.1f}%")

        data = {"avg_seq_len": avg_seq_len, "chain_sr": chain_sr, "task_info": task_info}
        # wandb.log({"avrg_performance/avg_seq_len": avg_seq_len, "avrg_performance/chain_sr": chain_sr, "detailed_metrics/task_info": task_info})
        current_data[epoch] = data

        print()
    previous_data = {}
    try:
        with open(log_dir / "results.json", "r") as file:
            previous_data = json.load(file)
    except FileNotFoundError:
        pass
    json_data = {**previous_data, **current_data}
    with open(log_dir / "results.json", "w") as file:
        json.dump(json_data, file, indent=2)
    print(f"Best model: epoch {max(ranking, key=ranking.get)} with average sequences length of {max(ranking.values())}")


def evaluate_policy(model, env, lang_embeddings, cfg, num_videos=0, save_dir=None):
    task_oracle = hydra.utils.instantiate(cfg.tasks)
    val_annotations = cfg.annotations

    # video stuff
    if num_videos > 0:
        rollout_video = RolloutVideo(
            logger=logger,
            empty_cache=False,
            log_to_file=True,
            save_dir=save_dir,
            resolution_scale=1,
        )
    else:
        rollout_video = None

    eval_sequences = get_sequences(cfg.num_sequences)

    results = []
    env_step_results = []
    plans = defaultdict(list)

    if not cfg.debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for i, (initial_state, eval_sequence) in enumerate(eval_sequences):
        record = i < num_videos
        # record = False

        result, env_step_dict = evaluate_sequence(
            env, model, task_oracle, initial_state, eval_sequence, lang_embeddings, val_annotations, cfg, record, rollout_video, i
        )

        results.append(result)
        env_step_results.append(env_step_dict)
        if record:
            rollout_video.write_to_tmp()
        if not cfg.debug:
            success_rates = count_success(results)
            average_rate = sum(success_rates) / len(success_rates) * 5
            description = " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(success_rates)])
            description += f" Average: {average_rate:.4f} |"
            eval_sequences.set_description(description)
        if record:
        # if result < 4 and record:
            rollout_video._log_currentvideos_to_file(i, save_as_video=True)

        # exit()
    return results, plans, env_step_results


def evaluate_sequence(
    env, model, task_checker, initial_state, eval_sequence, lang_embeddings, val_annotations, cfg, record, rollout_video, i
):
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    if record:
        caption = " | ".join(eval_sequence)
        rollout_video.new_video(tag=get_video_tag(i), caption=caption)
    success_counter = 0
    if cfg.debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    env_step_dict = {}
    for idx, subtask in enumerate(eval_sequence):
        if record:
            rollout_video.new_subtask()
        success, env_step_length = rollout_hi3(env, model, task_checker, cfg, subtask, lang_embeddings, val_annotations, record, rollout_video)
        env_step_dict[idx] = {
            'task': subtask,
            'env_step_length':env_step_length,
            'success': success
        }
        if record:
            rollout_video.draw_outcome(success)
        if success:
            success_counter += 1
        else:
            return success_counter, env_step_dict
    return success_counter, env_step_dict


def evaluate_sequence_retry(
    env, model, task_checker, initial_state, eval_sequence, lang_embeddings, val_annotations, cfg, record, rollout_video, i,
    retry_chance = 2
):
    max_length = 0
    for retry in range(retry_chance):
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        if record:
            caption = " | ".join(eval_sequence)
            rollout_video.new_video(tag=get_video_tag(i), caption=caption)
        success_counter = 0
        if cfg.debug:
            time.sleep(1)
            print()
            print()
            print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
            print("Subtask: ", end="")
        for subtask in eval_sequence:
            if record:
                rollout_video.new_subtask()
            success = rollout_hi3(env, model, task_checker, cfg, subtask, lang_embeddings, val_annotations, record, rollout_video)
            if record:
                rollout_video.draw_outcome(success)
            if success:
                success_counter += 1
            else:
                break
        max_length = max(success_counter, max_length)
        if max_length > 4 :
            break

    return max_length

def get_color(score):
    if score >= 0.05:      
        return (0, 255, 0)
    if score <= -0.05:     
        return (255, 0, 0)
    return (255, 255, 0)   

import torch.nn.functional as F
def rollout_hi3(env, model, task_oracle, cfg, subtask, lang_embeddings, val_annotations, record=False, rollout_video=None):
    
    # sample_num = cfg.sample_num
    if cfg.debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    # get language goal embedding
    goal = lang_embeddings.get_lang_goal(lang_annotation)
    goal['lang_text'] = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()

    env_step_length = 0

    for step in range(60):
        # print(f"chunk {step}")

        action_chunks = []

        # print("first_obs:", obs)
        # print("first_goal:", goal)

        sp, out1 = model.step(obs, goal)
        action_chunks.append(out1)
        # print("out1",out1.shape)
        obs, _, _, current_info = env.step(action_chunks[0][0])
        env_step_length+=1
        if cfg.debug:
            img = env.render(mode="rgb_array")
            join_vis_lang(img, lang_annotation)
            # time.sleep(0.1)
        if record:
            # update video
            rollout_video.update(obs["rgb_obs"]["rgb_static"])
            temp_sp = sp
            rollout_video.add_minmax(str(temp_sp)[:5], color = get_color(temp_sp))
        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if cfg.debug:
                print(colored("success", "green"), end=" ")
            if record:
                rollout_video.add_language_instruction(lang_annotation)
            return True, env_step_length
        


        ######################################################################################
        sp2, out2 = model.step(obs, goal)

        action_chunks.append(out2)
        temp_action = (action_chunks[0][1] + action_chunks[1][0])/2
        temp_action[-1].sign_()
        if temp_action[-1] == 0:
            temp_action[-1] = 1.0
        obs, _, _, current_info = env.step(temp_action)
        env_step_length+=1
        if cfg.debug:
            img = env.render(mode="rgb_array")
            join_vis_lang(img, lang_annotation)
            # time.sleep(0.1)
        if record:
            # update video
            rollout_video.update(obs["rgb_obs"]["rgb_static"])
            temp_sp = (sp+sp2)/2
            rollout_video.add_minmax(str(temp_sp)[:5], color = get_color(temp_sp))
        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if cfg.debug:
                print(colored("success", "green"), end=" ")
            if record:
                rollout_video.add_language_instruction(lang_annotation)
            return True, env_step_length


        ######################################################################################
        sp3, out3 = model.step(obs, goal)
        
        action_chunks.append(out3)

        for t in range(2, 10):
            action = (action_chunks[0][t] + action_chunks[1][t-1] + action_chunks[2][t-2]) / 3.0
            action[-1].sign_()
            if action[-1] == 0:
                action[-1] = 1.0
            obs, _, _, current_info = env.step(action)
            env_step_length+=1

            if cfg.debug:
                img = env.render(mode="rgb_array")
                join_vis_lang(img, lang_annotation)
                # time.sleep(0.1)
            if record:
                temp_sp = (sp+sp2+sp3)/3
                rollout_video.add_minmax(str(temp_sp)[:5], color = get_color(temp_sp))
                rollout_video.update(obs["rgb_obs"]["rgb_static"])
            # check if current step solves a task
            current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
            if len(current_task_info) > 0:
                if cfg.debug:
                    print(colored("success", "green"), end=" ")
                if record:
                    rollout_video.add_language_instruction(lang_annotation)
                return True, env_step_length
            
        ############################
        action = (action_chunks[1][-1] + action_chunks[2][-2]) / 2.0
        action[-1].sign_()
        if action[-1] == 0:
            action[-1] = 1.0
        obs, _, _, current_info = env.step(action)
        env_step_length+=1

        if cfg.debug:
            img = env.render(mode="rgb_array")
            join_vis_lang(img, lang_annotation)
            # time.sleep(0.1)
        if record:
            # update video
            temp_sp = (sp+sp2+sp3)/3
            rollout_video.add_minmax(str(temp_sp)[:5], color = get_color(temp_sp))
            rollout_video.update(obs["rgb_obs"]["rgb_static"])
        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if cfg.debug:
                print(colored("success", "green"), end=" ")
            if record:
                rollout_video.add_language_instruction(lang_annotation)
            return True, env_step_length
        

        ############################
        action = action_chunks[2][-1]
        action[-1].sign_()
        if action[-1] == 0:
            action[-1] = 1.0
        obs, _, _, current_info = env.step(action)
        env_step_length+=1

        if cfg.debug:
            img = env.render(mode="rgb_array")
            join_vis_lang(img, lang_annotation)
            # time.sleep(0.1)
        if record:
            # update video
            temp_sp = (sp+sp2+sp3)/3
            rollout_video.add_minmax(str(temp_sp)[:5], color = get_color(temp_sp))
            rollout_video.update(obs["rgb_obs"]["rgb_static"])
        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if cfg.debug:
                print(colored("success", "green"), end=" ")
            if record:
                rollout_video.add_language_instruction(lang_annotation)
            return True, env_step_length
        
    if cfg.debug:
        print(colored("fail", "red"), end=" ")
    if record:
        rollout_video.add_language_instruction(lang_annotation)
    return False, env_step_length


from typing import Optional, Union
from pathlib import Path
from dataclasses import dataclass

@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    # pretrained_checkpoint: Union[str, Path] = "../outputs/calvin-abc"     # Pretrained checkpoint path
    pretrained_checkpoint: Union[str, Path] = DEFAULT_VERIFIER_CHECKPOINT.as_posix()
    
    use_minivla: bool = False                   # If True, 

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_x0_prediction: bool = False
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = False                         # Whether to include proprio state in input

    center_crop: bool = False                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # CALVIN
    #################################################################################################################
    calvin_path: str = str(PROJECT_ROOT / "calvin")
    log_dir: str = str(PROJECT_ROOT / "logs")
    with_depth: bool = False
    with_gripper: bool = True
    with_cfg: bool = True
    enrich_lang: bool = False
    
    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                 # Random Seed (for reproducibility)

    # fmt: on
    save_version: str = "Pro"                        # version of exps


from experiments.robot.openvla_utils import (
    get_processor,
    get_proprio_projector,
    get_action_head_SP,
)
from experiments.robot.robot_utils import (
    get_model_sp
)
def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model_sp(cfg)
    model.set_version(cfg.save_version)
    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression:
        action_head = get_action_head_SP(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    return model, action_head, proprio_projector, noisy_action_projector, processor


#@hydra.main(config_path="../policy_conf", config_name="calvin_evaluate_all")
def main(cfg):

    


    log_wandb = cfg.log_wandb
    torch.cuda.set_device(cfg.device)
    seed_everything(cfg.seed, workers=True)  # type:ignore
    # evaluate a custom model
    lang_embeddings = None
    env = None
    results = {}
    plans = {}




    ##################
    sp_cfg = GenerateConfig
    sp_cfg.calvin_path = cfg.verifier_calvin_path
    sp_cfg.pretrained_checkpoint = cfg.verifier_checkpoint
    print(sp_cfg)
    sp_model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(sp_cfg)
    sp_model_wrap = DualSystemCalvinEvaluation_SP(sp_model, proprio_projector, noisy_action_projector, action_head, processor, use_x0_prediction=sp_cfg.use_x0_prediction)
    sp_model_wrap.reset()
    ###################

    print('train_folder',cfg.train_folder)
    if not cfg.steps:
        checkpoints = get_all_checkpoints(Path(cfg.train_folder))
    else:
        checkpoints = get_all_checkpoints(Path(cfg.train_folder))

        if not '+' in cfg.steps:
            print("all checkpoints:", checkpoints)
            checkpoints = [chk for chk in checkpoints if str(chk).split('/')[-1].startswith(f"{cfg.steps}")]
        else:
            eval_steps = cfg.steps.split('+')
            print('eval_steps:',eval_steps)
            newcheckpoints = []
            for chk in checkpoints:
                # print(chk)
                for estep in eval_steps:
                    if str(chk).split('/')[-1].startswith(f"{estep}"):
                        newcheckpoints.append(chk)
                        break
            checkpoints = newcheckpoints
    print("find checkpoints:", checkpoints)
    # aaa
    for checkpoint in checkpoints:
        print(cfg.device)
        env, _, lang_embeddings = get_default_beso_and_env(
            cfg.train_folder,
            cfg.root_data_dir,
            checkpoint,
            env=env,
            lang_embeddings=lang_embeddings,
            eval_cfg_overwrite=cfg.eval_cfg_overwrite,
            device_id=cfg.device,
            cfg=cfg,
        )

        ckpt_path = cfg.train_folder
        ckpt = os.path.join(ckpt_path, checkpoint)

        print(f"Loading model from {ckpt}")
        state_dict = torch.load(ckpt, map_location='cpu')
        device = torch.device(f"cuda:{cfg.device}")
        model = hydra.utils.instantiate(cfg.model)
        model.load_state_dict(state_dict['model'],strict = False)
        model.freeze()
        model = model.cuda(device)
        model.num_sampling_steps = cfg.num_sampling_steps
        model.sampler_type = cfg.sampler_type
        model.multistep = cfg.multistep

        if cfg.sigma_min is not None:
            model.sigma_min = cfg.sigma_min
        if cfg.sigma_max is not None:
            model.sigma_max = cfg.sigma_max
        if cfg.noise_scheduler is not None:
            model.noise_scheduler = cfg.noise_scheduler

        if cfg.cfg_value != 1:
            raise NotImplementedError("cfg_value != 1 not implemented yet")
        model.process_device()
        model.eval()

        if log_wandb:
            log_dir = get_log_dir(cfg.eval_save_dir)
            os.makedirs(log_dir / "wandb", exist_ok=False)

            from policy_evaluation.sampling_wrapper import SamplingFramework
            
            print("type env", type(env))

            observation_space = {'rgb_obs': ['rgb_static', 'rgb_gripper'], 'depth_obs': [], 'state_obs': ['robot_obs'], 'actions': ['rel_actions'], 'language': ['language']}

            framework = SamplingFramework(policy = model,
                                          verifier = sp_model_wrap,
                                          
                                          same_transform = True,
                                          policy_transform = env.transforms,
                                          verifier_transform = env.transforms,
                                          sample_config = {
                                                    'sample_type': cfg.sample_type,
                                                    'policy_sample_num': cfg.policy_sample_num,
                                                    'total_sample_num': cfg.total_sample_num,
                                                    'max_batch_num': cfg.max_batch_num,
                                                },
                                          selection_config = {
                                                    'selection_type': cfg.selection_type,
                                                    'selection_direction': cfg.selection_direction,
                                                    'selection_threshold': cfg.selection_threshold,
                                                },
                                          device = env.device,
                                          proprio_state = env.proprio_state,
                                          observation_space = observation_space
                            )

            results[checkpoint], plans[checkpoint], env_step_results = evaluate_policy(framework, env, lang_embeddings, cfg, num_videos=cfg.num_videos, save_dir=Path(log_dir))
            print_and_save(results, plans, cfg, log_dir=log_dir)
            env_step_length_statistic(env_step_results)

            import json
            lst = results[checkpoint]
            data = {idx: val for idx, val in enumerate(lst)}
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, f"results_{str(checkpoint).split('/')[-1].split('.')[0]}.json"), "w") as f:
                json.dump(data, f, indent=4)

def env_step_length_statistic(results):

    results_success = {}
    results_faied = {}  
    results_all = {}


    for sequence_idx, env_step_results in enumerate(results):
        print("sequence_idx", sequence_idx, env_step_results)

        for subtask_idx in range(5):
            if not subtask_idx in env_step_results.keys():
                # print(f"{subtask_idx} not in keys", env_step_results[subtask_idx].keys())
                break
            task = env_step_results[subtask_idx]['task']
            env_step_length = env_step_results[subtask_idx]['env_step_length']
            success = env_step_results[subtask_idx]['success']
            
            if success:
                if not task in results_success.keys():
                    results_success[task] = [env_step_length]
                else:
                    results_success[task].append(env_step_length)

                if not task in results_all.keys():
                    results_all[task] = [env_step_length]
                else:
                    results_all[task].append(env_step_length)

            if not success:
                if not task in results_faied.keys():
                    results_faied[task] = [env_step_length]
                else:
                    results_faied[task].append(env_step_length)

                if not task in results_all.keys():
                    results_all[task] = [env_step_length]
                else:
                    results_all[task].append(env_step_length)

    print("results_success", results_success)
    print("results_failed", results_faied)
    print("results_all", results_all)
    print("success:")
    for subtask_name, all_length in sorted(results_success.items()):
        print(subtask_name, np.mean(all_length))

    print("failed:")
    for subtask_name, all_length in sorted(results_faied.items()):
        print(subtask_name, np.mean(all_length))

    print("all:")
    for subtask_name, all_length in sorted(results_all.items()):
        print(subtask_name, np.mean(all_length))


if __name__ == "__main__":
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    # Set CUDA device IDs
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_model_path",
        type=str,
        default=os.environ.get("VIDEO_MODEL_PATH", str(PROJECT_ROOT / "official_checkpoints" / "svd-robot-calvin-ft")),
    )
    parser.add_argument("--action_model_folder", type=str, default=os.environ.get("ACTION_MODEL_FOLDER", ""))
    parser.add_argument(
        "--clip_model_path",
        type=str,
        default=os.environ.get("CLIP_MODEL_PATH", str(PROJECT_ROOT / "official_checkpoints" / "clip-vit-base-patch32")),
    )
    parser.add_argument(
        "--calvin_abc_dir",
        type=str,
        default=os.environ.get("CALVIN_DATASET_ROOT", str(PROJECT_ROOT / "calvin" / "dataset" / "task_ABC_D")),
    )
    parser.add_argument(
        "--verifier_calvin_path",
        type=str,
        default=str(PROJECT_ROOT / "calvin"),
    )
    parser.add_argument(
        "--verifier_checkpoint",
        type=str,
        default=DEFAULT_VERIFIER_CHECKPOINT.as_posix(),
    )
    parser.add_argument(
        "--eval_save_dir",
        type=str,
        default=str(PROJECT_ROOT / "evaluation"),
    )
    parser.add_argument("--sample_type", type=str, default="vae", choices=["policy", "vae"])
    parser.add_argument("--policy_sample_num", type=int, default=4)
    parser.add_argument("--total_sample_num", type=int, default=24)
    parser.add_argument("--max_batch_num", type=int, default=4)
    parser.add_argument(
        "--selection_type",
        type=str,
        default="thres_weighted_merge",
        choices=["greedy", "thres_weighted_merge", "thres_minimal_3"],
    )
    parser.add_argument("--selection_direction", type=str, default="good", choices=["good", "bad"])
    parser.add_argument("--selection_threshold", type=float, default=0.08)
    parser.add_argument("--steps", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_config", type=str, required=True)

    args = parser.parse_args()

    with initialize_config_dir(config_dir=str(POLICY_CONF_DIR), job_name="job_name"):
        cfg = compose(config_name="calvin_evaluate_all.yaml")

    with initialize_config_dir(config_dir=str(POLICY_CONF_DIR), job_name="job_name"):
        cfg_for_model = compose(config_name=args.train_config)

    cfg.model = cfg_for_model.model


    cfg.model.pretrained_model_path = args.video_model_path
    cfg.train_folder = args.action_model_folder
    cfg.model.text_encoder_path = args.clip_model_path
    cfg.root_data_dir = args.calvin_abc_dir
    cfg.seed = args.seed
    OmegaConf.set_struct(cfg, False)
    cfg.steps = args.steps
    cfg.verifier_calvin_path = args.verifier_calvin_path
    cfg.verifier_checkpoint = args.verifier_checkpoint
    cfg.eval_save_dir = args.eval_save_dir
    cfg.sample_type = args.sample_type
    cfg.policy_sample_num = args.policy_sample_num
    cfg.total_sample_num = args.total_sample_num
    cfg.max_batch_num = args.max_batch_num
    cfg.selection_type = args.selection_type
    cfg.selection_direction = args.selection_direction
    cfg.selection_threshold = args.selection_threshold
    OmegaConf.set_struct(cfg, True)

    print(OmegaConf.to_yaml(cfg))

    main(cfg)
