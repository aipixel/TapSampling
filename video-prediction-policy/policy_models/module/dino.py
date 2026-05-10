# from typing import Dict, Optional, Tuple, Union
# from diffusers.models import UNetSpatioTemporalConditionModel
# from diffusers import TextToVideoSDPipeline, StableVideoDiffusionPipeline
# import torch
# import torch.nn as nn
# from einops import rearrange, repeat
# import math
# import random
# from transformers import AutoTokenizer, CLIPTextModelWithProjection
# import numpy as np

import torch
import torch.nn as nn


class Dino(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        local_repo = "/home/sizhe/clear_code/video-prediction-policy/external_toolkits/dinov2"
        ckpt_path  = "/home/sizhe/clear_code/video-prediction-policy/official_checkpoints/dinov2_vitl14_reg4_pretrain.pth"

        model = torch.hub.load(local_repo, "dinov2_vitl14_reg", source="local", pretrained=False)

        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

        clean_state = { (k[len("module."): ] if k.startswith("module.") else k): v for k, v in state_dict.items() }
        model.load_state_dict(clean_state, strict=False)

        self.model = model


    def forward(
        self,
        img
    ):
        """
        perceptual_features = self.TVP_encoder(input_rgb, language, self.timestep,                  # 扩散生成视频，获得特征
                                                            self.extract_layer_idx, all_layer=self.use_all_layer,
                                                            step_time=1, max_length=self.max_length)

        perceptual_features:   b f c h w        
        b,16,?,?,? 
        b f c (h w)
        b f l c     然后static、gripper在l上拼接
        """
        with torch.no_grad():
            feats = self.model.forward_features(img)
        """
        x_norm_clstoken torch.Size([4, 1024])
        x_norm_regtokens torch.Size([4, 4, 1024])
        x_norm_patchtokens torch.Size([4, 1369, 1024])
        x_prenorm torch.Size([4, 1374, 1024])
        """
        return feats["x_norm_patchtokens"]
    



if __name__ == "__main__":
    model = Dino()
    input = torch.zeros(4, 3, 518, 518)
    feats = model(input)

    for k, v in feats.items():
        if v is not None:
            print(k, v.shape)
