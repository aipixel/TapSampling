import logging
from typing import Dict, Optional, Tuple
from functools import partial
from torch import einsum, nn
import copy
from einops import rearrange, repeat
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import einops
from policy_models.edm_diffusion.score_wrappers import GCDenoiser
from policy_models.module.transformers.action_head import L1RegressionActionHead_Modified

from policy_models.module.clip_lang_encoder import LangClip
from policy_models.edm_diffusion.gc_sampling import *
from policy_models.utils.lr_schedulers.tri_stage_scheduler import TriStageLRScheduler
from policy_models.module.Video_Former import Video_Former_2D,Video_Former_3D
from diffusers import StableVideoDiffusionPipeline
from policy_models.module.diffusion_extract import Diffusion_feature_extractor
from transformers import AutoTokenizer, CLIPTextModelWithProjection


logger = logging.getLogger(__name__)

def load_primary_models(pretrained_model_path, eval=False):
    if eval:
        pipeline = StableVideoDiffusionPipeline.from_pretrained(pretrained_model_path, torch_dtype=torch.float16)
    else:
        pipeline = StableVideoDiffusionPipeline.from_pretrained(pretrained_model_path)
    return pipeline, None, pipeline.feature_extractor, pipeline.scheduler, pipeline.video_processor, \
        pipeline.image_encoder, pipeline.vae, pipeline.unet


class VPP_Adapter(pl.LightningModule):
    """
    The lightning module used for training.
    """

    def __init__(
            self,
            optimizer: DictConfig,
            lr_scheduler: DictConfig,
            latent_dim: int = 512,
            multistep: int = 10,
            use_lr_scheduler: bool = True,
            act_window_size: int = 10,
            use_text_not_embedding: bool = False,
            seed: int = 42,
            pretrained_model_path: str = '/cephfs/shared/gyj/ckpt/svd_pre/checkpoint-100000',
            text_encoder_path: str = '/home/disk2/gyj/hyc_ckpt/llm/clip-vit-base-patch32',
            use_position_encoding: bool = True,
            Former_depth: int = 3,
            Former_heads: int = 8,
            Former_dim_head: int = 64,
            Former_num_time_embeds: int = 1,
            num_latents: int = 3,
            use_Former: str = '3d',
            timestep: int = 20,
            max_length: int = 20,
            extract_layer_idx: int = 1,
            use_all_layer: bool = False,
            obs_seq_len: int = 1,
            action_dim: int = 7,
            action_seq_len: int = 10,

            train_svd: bool = False,
            inner_model_depth = 3
    ):
        super(VPP_Adapter, self).__init__()

        self.Former_depth = Former_depth
        self.inner_model_depth = inner_model_depth
        assert inner_model_depth % Former_depth == 0, "inner_model_depth mod Former_depth shoud be 0"
        # self.former_to_inner = int(inner_model_depth / Former_depth)

        self.latent_dim = latent_dim
        self.use_all_layer = use_all_layer
        self.use_position_encoding = use_position_encoding

        self.act_window_size = act_window_size
        self.action_dim = action_dim

        self.timestep = timestep
        self.extract_layer_idx = extract_layer_idx
        self.use_Former = use_Former
        self.Former_num_time_embeds = Former_num_time_embeds
        self.max_length = max_length

        condition_dim_list = [1280,1280,1280,640]
        sum_dim = 0
        for i in range(extract_layer_idx+1):
            sum_dim = sum_dim + condition_dim_list[i+1]
        condition_dim = condition_dim_list[extract_layer_idx+1] if not self.use_all_layer else sum_dim

        if use_Former=='3d':
            self.Video_Former = Video_Former_3D(
                dim=latent_dim,
                depth=Former_depth,
                dim_head=Former_dim_head,
                heads=Former_heads,
                num_time_embeds=Former_num_time_embeds,
                num_latents=num_latents,
                condition_dim=condition_dim,
                use_temporal=True,

                return_all_layers = True
             )
        
        print('use_Former:', self.use_Former)
        print('use_all_layer',self.use_all_layer)

        self.seed = seed
        self.use_lr_scheduler = use_lr_scheduler
        # goal encoders
        self.language_goal = LangClip(model_name='ViT-B/32').to(self.device)

        pipeline, tokenizer, feature_extractor, train_scheduler, vae_processor, text_encoder, vae, unet = load_primary_models(
            pretrained_model_path , eval = True)
        print("text_encoder_path")
        text_encoder = CLIPTextModelWithProjection.from_pretrained(text_encoder_path)
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_path, use_fast=False)

        text_encoder = text_encoder.to(self.device).eval()

        for param in pipeline.image_encoder.parameters():
            param.requires_grad = False
        for param in text_encoder.parameters():
            param.requires_grad = False

        for param in pipeline.vae.parameters():
            param.requires_grad = False

        self.train_svd = train_svd
        if self.train_svd:
            for name, param in pipeline.unet.named_parameters():
                # print(name, param.shape, param.requires_grad)
                param.requires_grad = True
            pipeline.unet.train()
        else:
            for param in pipeline.unet.parameters():
                param.requires_grad = False
            pipeline.unet.eval()

        pipeline = pipeline.to(self.device)

        self.TVP_encoder = Diffusion_feature_extractor(pipeline=pipeline,
                                                        tokenizer=tokenizer,
                                                        text_encoder=text_encoder,
                                                        position_encoding = self.use_position_encoding)
        self.TVP_encoder = self.TVP_encoder.to(self.device)
        self.add_module("pipeline_unet", pipeline.unet)
        # self.pipeline_unet = self.TVP_encoder.pipeline.unet

        # policy network
        self.model = L1RegressionActionHead_Modified(input_dim = latent_dim,
                                                     hidden_dim = latent_dim,
                                                     action_dim = action_dim,
                                                     use_pro_version = True,
                                                     action_length = action_seq_len,
                                                     depth = inner_model_depth,
                                                     raw_feature_dim = condition_dim
                                                     )

        self.optimizer_config = optimizer
        self.lr_scheduler = lr_scheduler
        self.save_hyperparameters()
        # for inference
        self.rollout_step_counter = 0
        self.multistep = multistep
        self.latent_goal = None
        self.plan = None
        self.use_text_not_embedding = use_text_not_embedding
        self.ema_callback_idx = None

    def process_device(self):
        self.TVP_encoder.pipeline = self.TVP_encoder.pipeline.to(self.device)
        self.TVP_encoder.text_encoder = self.TVP_encoder.text_encoder.to(self.device)

    def configure_optimizers(self):
        """
        Initialize optimizers and learning rate schedulers based on model configuration.
        """
        # Configuration for models using transformer weight decay
        '''optim_groups = self.action_decoder.model.inner_model.get_optim_groups(
            weight_decay=self.optimizer_config.transformer_weight_decay
        )'''

        def get_by_path(obj, path):
            """获取嵌套属性"""
            cur = obj
            for part in path.split('.'):
                if not hasattr(cur, part):
                    return None
                cur = getattr(cur, part)
            return cur
        
        modules = [
            "Video_Former",
            "pipeline_unet"
            # "TVP_encoder.pipeline.unet",
        ]

        param_lists = []             # 用于传给 optimizer 的参数列表（扁平）
        trainable_names = []         # 需要梯度的参数名（字符串）
        frozen_names = []            # 不需要梯度的参数名
        seen_param_ids = set()       # 去重用（基于 id(param)）

        for mod_name in modules:
            module = get_by_path(self, mod_name)
            if module is None:
                continue

            mod_trainable_params = []
            for pname, p in module.named_parameters(recurse=True):
                full_name = f"{mod_name}.{pname}"
                pid = id(p)
                if pid in seen_param_ids:
                    print(f"pid {pid} exist, {full_name}")
                    continue
                seen_param_ids.add(pid)

                if p.requires_grad:
                    mod_trainable_params.append(p)
                    trainable_names.append(full_name)
                else:
                    frozen_names.append(full_name)

            if mod_trainable_params:
                param_lists.extend(mod_trainable_params)
                print(f"add {len(mod_trainable_params)} params from '{mod_name}' to optimizer list")

        # print(f"Total trainable param names: {len(trainable_names)}")
        # print(f"Total frozen param names: {len(frozen_names)}")

        # print("trainable", trainable_names)
        # print("frozen:", frozen_names)

        optim_groups = [
            {"params": self.model.parameters(),
             "weight_decay": self.optimizer_config.transformer_weight_decay},
            {"params": param_lists, 
             "weight_decay": self.optimizer_config.transformer_weight_decay},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.optimizer_config.learning_rate,
                                      betas=self.optimizer_config.betas)

        # Optionally initialize the scheduler
        if self.use_lr_scheduler:
            lr_configs = OmegaConf.create(self.lr_scheduler)
            scheduler = TriStageLRScheduler(optimizer, lr_configs)
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": 'step',
                "frequency": 1,
            }
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        else:
            return optimizer

    def on_before_zero_grad(self, optimizer=None):
        total_grad_norm = 0.0
        total_param_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.norm().item() ** 2
            total_param_norm += p.norm().item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        total_param_norm = total_param_norm ** 0.5

        self.log("train/grad_norm", total_grad_norm, on_step=True, on_epoch=False, sync_dist=True)
        self.log("train/param_norm", total_param_norm, on_step=True, on_epoch=False, sync_dist=True)


    def training_step(self, dataset_batch: Dict[str, Dict],) -> torch.Tensor:  # type: ignore
        total_loss, action_loss = (
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
        )
        predictive_feature, latent_goal= self.extract_predictive_feature(dataset_batch)
        #'state_images'
        #'raw_features'

        # for idx, what in enumerate(predictive_feature["state_images"]):
        #     print(idx, what.shape)
        # print("raw_features", predictive_feature["raw_features"].shape)
        # aaa

        action_pred = self.model.predict_action(actions_hidden_states = predictive_feature['state_images'], # from videoformer, (b, 224, 384)
                                 proprio = None,
                                 proprio_projector = None,
                                 task_hidden_states = predictive_feature['raw_features'],    # from SVD, (b, 512, 2560)
                                 phase="Training"
                                 ) 

        # print("action_pred", action_pred.shape)  
        # aaa

        target_actions = dataset_batch["actions"]
        act_loss = nn.L1Loss()(target_actions, action_pred)

        action_loss += act_loss
        total_loss += act_loss

        total_bs = dataset_batch["actions"].shape[0]
        self._log_training_metrics(action_loss, total_loss, total_bs)
        return total_loss

    @torch.no_grad()
    def validation_step(self, dataset_batch: Dict[str, Dict]) -> Dict[
        str, torch.Tensor]:  # type: ignore
        """
        Compute and log the validation losses and additional metrics.
        During the validation step, the diffusion model predicts the next action sequence given the current state

        Args:
            batch: Dictionary containing the batch data for each modality.
            batch_idx: Index of the batch. used for compatibility with pytorch lightning.
            dataloader_idx: Index of the dataloader. used for compatibility with pytorch lightning.

        Returns:
            Dictionary containing the sampled plans of plan recognition and plan proposal module, as well as the
            episode indices.
        """

        # 没有调试这部分代码，因为我没写validation

        output = {}
        val_total_act_loss_pp = torch.tensor(0.0).to(self.device)
            # Compute the required embeddings
        predictive_feature, latent_goal= self.extract_predictive_feature(dataset_batch)
        #'state_images'
        #'raw_features'

        action_pred = self.model.predict_action(actions_hidden_states = predictive_feature['state_images'], # from videoformer, (b, 224, 384)
                                 proprio = None,
                                 proprio_projector = None,
                                 task_hidden_states = predictive_feature['raw_features'],    # from SVD, (b, 512, 2560)
                                 phase="Inference"
                                 )    

        dataset_batch["actions"] = dataset_batch["actions"].to(action_pred.device)
        # compute the mse action loss
        pred_loss = torch.nn.functional.mse_loss(action_pred, dataset_batch["actions"])
        val_total_act_loss_pp += pred_loss

        output[f"idx:"] = dataset_batch["idx"]
        output["validation_loss"] = val_total_act_loss_pp
        return output

    def extract_predictive_feature(self, dataset_batch):
        """
        Compute the required embeddings for the visual ones and the latent goal.
        """
        # 1. extract the revelant visual observations
        rgb_static = dataset_batch["rgb_obs"]['rgb_static'].to(self.device)
        rgb_gripper = dataset_batch["rgb_obs"]['rgb_gripper'].to(self.device)
        # 3. we compute the language goal if the language modality is in the scope
        modality = "lang"
        if self.use_text_not_embedding:
            latent_goal = self.language_goal(dataset_batch["lang_text"]).to(rgb_static.dtype)
        else:
            latent_goal = self.language_goal(dataset_batch["lang"]).to(rgb_static.dtype)

        language = dataset_batch["lang_text"]

        num_frames = self.Former_num_time_embeds
        rgb_static = rgb_static.to(self.device)
        rgb_gripper = rgb_gripper.to(self.device)
        batch = rgb_static.shape[0]

        with torch.no_grad():
            input_rgb = torch.cat([rgb_static, rgb_gripper], dim=0)                         # b*2, 1, 3, 256, 256   # rgb+gripper
            language = language + language
            perceptual_features = self.TVP_encoder(input_rgb, language, self.timestep,                  # 扩散生成视频，获得特征
                                                           self.extract_layer_idx, all_layer=self.use_all_layer,
                                                           step_time=1, max_length=self.max_length)

        perceptual_features = einops.rearrange(perceptual_features, 'b f c h w-> b f c (h w)')
        perceptual_features = einops.rearrange(perceptual_features, 'b f c l-> b f l c')
        perceptual_features = perceptual_features[:, :num_frames, :, :]
        #print('perceptual_features_shape:', perceptual_features.shape)
        perceptual_features, gripper_feature = torch.split(perceptual_features, [batch, batch], dim=0)
        perceptual_features = torch.cat([perceptual_features, gripper_feature], dim=2)

        perceptual_features = perceptual_features.to(torch.float32)
        # print("1 perceptual_features before Video_Former", perceptual_features.shape)     # b, 16, 512, 2560


        raw_features = perceptual_features[:, 0, ...]   # b, 512, 2560      #第一帧，静态
        perceptual_features = self.Video_Former(perceptual_features)        #特征编码
        # print("2 perceptual_features after Video_Former", perceptual_features.shape)      # List: N × (b, 224, D)
        predictive_feature = {'state_images': perceptual_features,
                              'raw_features': raw_features}         # b,512,2560
        predictive_feature['modality'] = modality
        return predictive_feature, latent_goal


    def _log_training_metrics(self, action_loss, total_loss, total_bs):
        """
        Log the training metrics.
        """
        self.log("train/action_loss", action_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)
        self.log("train/total_loss", total_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=total_bs)

    def _log_validation_metrics(self, pred_loss, img_gen_loss, val_total_act_loss_pp):
        """
        Log the validation metrics.
        """
        self.log(
            "val_act/action_loss",
            val_total_act_loss_pp / len(self.trainer.datamodule.modalities),  # type:ignore
            sync_dist=True,
        )
        self.log(f"val_act/img_gen_loss_pp", img_gen_loss, sync_dist=True)

    def reset(self):
        """
        Call this at the beginning of a new rollout when doing inference.
        """
        self.plan = None
        self.latent_goal = None
        self.rollout_step_counter = 0

    def forward(self,batch):
        return self.training_step(batch)
        #def training_step(self, batch: Dict[str, Dict], batch_idx: int,
        #                  dataloader_idx: int = 0) -> torch.Tensor

    def eval_forward(self, obs, goal,
                     sample_num = 1):
        """
        Method for doing inference with the model.
        """
        if 'lang_text' in goal:
            if self.use_text_not_embedding:
                # print(goal.keys())
                latent_goal = self.language_goal(goal["lang_text"])
                latent_goal = latent_goal.to(torch.float32)
            else:
                latent_goal = self.language_goal(goal["lang"]).unsqueeze(0).to(torch.float32).to(
                    obs["rgb_obs"]['rgb_static'].device)

        rgb_static = obs["rgb_obs"]['rgb_static']
        # rgb_gripper = dataset_batch["rgb_obs"]['rgb_gripper'][:, :-1]
        rgb_gripper = obs["rgb_obs"]['rgb_gripper']

        language = goal["lang_text"]

        num_frames = self.Former_num_time_embeds

        rgb_static = rgb_static.to(self.device)
        rgb_gripper = rgb_gripper.to(self.device)


        rgb_static = rgb_static.repeat(sample_num, 1, 1, 1, 1)      # (1,1,3,256,256) -> (N,1,3,256,256)
        rgb_gripper = rgb_gripper.repeat(sample_num, 1, 1, 1, 1)      # (1,1,3,256,256) -> (N,1,3,256,256)
        batch = rgb_static.shape[0]     # N

        with torch.no_grad():
            input_rgb = torch.cat([rgb_static, rgb_gripper], dim=0)     # ([2N, 1, 3, 256, 256])
            # language = [language] + [language]
            language = [copy.deepcopy(language) for _ in range(input_rgb.shape[0])]     # 2N

            perceptual_features = self.TVP_encoder.forward(input_rgb, language, self.timestep,
                                                           self.extract_layer_idx, all_layer=self.use_all_layer,
                                                           step_time=1, max_length=self.max_length)
            
        perceptual_features = einops.rearrange(perceptual_features, 'b f c h w-> b f c (h w)')
        perceptual_features = einops.rearrange(perceptual_features, 'b f c l-> b f l c')
        perceptual_features = perceptual_features[:, :num_frames, :, :]

        perceptual_features, gripper_feature = torch.split(perceptual_features, [batch, batch], dim=0)
        perceptual_features = torch.cat([perceptual_features, gripper_feature], dim=2)

        perceptual_features = perceptual_features.to(torch.float32)

        raw_features = perceptual_features[:, 0, ...]   # b, 512, 2560      #第一帧，静态
        perceptual_features = self.Video_Former(perceptual_features)
        if self.use_Former == 'linear':
            perceptual_features = rearrange(perceptual_features, 'b T q d -> b (T q) d')

        perceptual_emb = {'state_images': perceptual_features,
                          'raw_features': raw_features}

        perceptual_emb['modality'] = "lang"

        action_pred = self.model.predict_action(actions_hidden_states = perceptual_emb['state_images'], # from videoformer, (b, 224, 384)
                                 proprio = None,
                                 proprio_projector = None,
                                 task_hidden_states = perceptual_emb['raw_features'],    # from SVD, (b, 512, 2560)
                                 phase="Inference"
                                 )
        
        return action_pred


    def step(self, obs, goal, 
             return_chunk = False,
             sample_num = 1):

        if self.rollout_step_counter % self.multistep == 0:
            pred_action_seq = self.eval_forward(obs, goal,
                                                sample_num = sample_num)
            self.pred_action_seq = pred_action_seq

        if return_chunk and sample_num==1:
            self.rollout_step_counter = 0
            return pred_action_seq[0]               # 10, 7
        if return_chunk and sample_num > 1:
            self.rollout_step_counter = 0           # b, 10, 7
            return pred_action_seq

        raise NotImplementedError("")
    
    
    def on_train_start(self) -> None:

        self.model.to(dtype=self.dtype)

        self.Video_Former.to(dtype=self.dtype)
        self.language_goal.to(dtype=self.dtype)
        #self.vae.to(dtype=self.dtype)
        self.TVP_encoder.to(dtype=self.dtype)

    @rank_zero_only
    def on_train_epoch_start(self) -> None:
        logger.info(f"Start training epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_end(self, unused: Optional = None) -> None:  # type: ignore
        logger.info(f"Finished training epoch {self.current_epoch}")

    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        logger.info(f"Finished validation epoch {self.current_epoch}")


    def on_validation_epoch_start(self) -> None:
        log_rank_0(f"Start validation epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_start(self) -> None:
        logger.info(f"Start training epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_end(self, unused: Optional = None) -> None:  # type: ignore
        logger.info(f"Finished training epoch {self.current_epoch}")

    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        logger.info(f"Finished validation epoch {self.current_epoch}")

    def on_validation_epoch_start(self) -> None:
        log_rank_0(f"Start validation epoch {self.current_epoch}")

@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)