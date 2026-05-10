# model_server.py  —— 独立进程脚本
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uvicorn
import os
import logging
import traceback
import torch
import numpy as np
from threading import Lock

from typing import Any, Optional
import draccus

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=2)
MODEL = None

MODEL_LOCK = Lock()



class StepRequest(BaseModel):
    obs: Any
    instruction: Any
    step: int
    query_actions: Optional[Any] = None

class StepResponse(BaseModel):
    result: Any



@app.post("/step", response_model=StepResponse)
async def step_endpoint(req: StepRequest):
    global MODEL
    if MODEL is None:
        raise HTTPException(status_code=503, detail="model not ready")
    loop = asyncio.get_running_loop()
    try:

        print("get request with instruction", req.instruction)

        result = await loop.run_in_executor(
            executor,
            _call_step,
            req.obs,
            req.instruction,
            req.step,
            req.query_actions
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model execution error: {e}")
    return StepResponse(result=result)

def _get_model_device(model):
    try:
        for p in model.parameters():
            return p.device
    except Exception:
        pass
    return None

def _to_torch_recursive(x, device=None):
    # already torch
    if torch is not None and isinstance(x, torch.Tensor):
        return x.to(device) if device is not None else x
    # numpy array
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
        return t.to(device) if device is not None else t
    # list/tuple -> try to make tensor (unless elements are dicts)
    if isinstance(x, (list, tuple)):
        # if elements are dict or list of dicts, convert elementwise
        if len(x) > 0 and any(isinstance(el, dict) for el in x):
            return [ _to_torch_recursive(el, device) for el in x ]
        try:
            t = torch.tensor(x)
            return t.to(device) if device is not None else t
        except Exception:
            # fallback to elementwise conversion
            return [ _to_torch_recursive(el, device) for el in x ]
    # dict -> convert each value
    if isinstance(x, dict):
        return {k: _to_torch_recursive(v, device) for k, v in x.items()}
    # numpy scalar
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()
    # basic types unchanged
    return x

def _serialize_result_recursive(res):
    # torch tensor
    if torch is not None and isinstance(res, torch.Tensor):
        return res.detach().cpu().numpy().tolist()
    # numpy array
    if isinstance(res, np.ndarray):
        return res.tolist()
    # numpy scalar
    if isinstance(res, (np.integer, np.floating, np.bool_)):
        return res.item()
    # dict
    if isinstance(res, dict):
        return {k: _serialize_result_recursive(v) for k, v in res.items()}
    # list/tuple
    if isinstance(res, (list, tuple)):
        return [ _serialize_result_recursive(v) for v in res ]
    # basic types (int/float/str/bool/None) - should be JSON serializable already
    try:
        import json
        json.dumps(res)
        return res
    except Exception:
        # fallback to string
        return str(res)

def _call_step(obs, instruction, step_idx, query_actions):
    try:
        device = _get_model_device(MODEL)

        print("model device:", device)

        obs_in = {
            'rgb_obs': {
                'rgb_static': np.array(obs['rgb_obs']["rgb_static"]).astype(np.uint8),
                'rgb_gripper': np.array(obs['rgb_obs']["rgb_gripper"]).astype(np.uint8)
            }
        }

        # print(obs_in['rgb_obs']['rgb_static'].shape)
        # print(obs_in['rgb_obs']['rgb_static'])

        qa_in = torch.from_numpy(np.array(query_actions)).to(torch.bfloat16).cuda()

        if MODEL_LOCK:
            MODEL_LOCK.acquire()
        try:
            if torch is not None:
                with torch.no_grad():
                    out = MODEL.step(obs_in, instruction, step_idx, query_actions=qa_in)
            else:
                out = MODEL.step(obs_in, instruction, step_idx, query_actions=qa_in)
        finally:
            if MODEL_LOCK:
                MODEL_LOCK.release()


        print("raw scores", out.shape)
        

        out = out.squeeze(-1).mean(-1)
        print("get output!:", out)

        return _serialize_result_recursive(out)

    except Exception:
        traceback.print_exc()
        raise


###################################################################################################

from vla_evaluation import DualSystemCalvinEvaluation_SP


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
    pretrained_checkpoint: Union[str, Path] = "/home/sizhe/clear_code/tapsampling/output/CALVIN-ABC-Pro"
    
    use_minivla: bool = False                   # If True, 

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_x0_prediction: bool = False
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = False                         # Whether to include proprio state in input
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization
    center_crop: bool = False                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy
    unnorm_key: Union[str, Path] = ""                # Action un-normalization key


    #################################################################################################################
    # CALVIN
    #################################################################################################################
    calvin_path: str = "calvin"
    log_dir: str = "log"
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

import sys
from pathlib import Path

TAPSAMPLING_ROOT = Path(__file__).resolve().parents[1]
if TAPSAMPLING_ROOT.as_posix() not in sys.path:
    sys.path.append(TAPSAMPLING_ROOT.as_posix())

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

    noisy_action_projector = None

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def _reenable_uvicorn_loggers() -> None:
    # prismatic.overwatch config disables existing loggers globally.
    # Re-enable uvicorn's loggers so startup and access logs remain visible.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).disabled = False


@draccus.wrap()
def main(cfg: GenerateConfig) -> None:
    global MODEL

    port = int(os.environ.get("PORT", 8002))
    gpu_id = os.environ.get("GPU_ID", "")
    if gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        print("Server_Config:", cfg)
        sp_model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
        print("Init model success")
        sp_model_wrap = DualSystemCalvinEvaluation_SP(
            sp_model,
            proprio_projector,
            noisy_action_projector,
            action_head,
            processor,
            use_x0_prediction=cfg.use_x0_prediction,
        )
        sp_model_wrap.reset()
        MODEL = sp_model_wrap
        print("Load model success")

    except Exception:
        traceback.print_exc()
        raise

    _reenable_uvicorn_loggers()
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")



if __name__ == "__main__":
    main()
