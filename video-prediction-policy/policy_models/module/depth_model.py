import torch
import torch.nn as nn
import copy
from typing import Optional, Dict, Any
from diffusers import UNetSpatioTemporalConditionModel

class Depth_UNet(nn.Module):
    """
    Wrapper around UNetSpatioTemporalConditionModel built from a config dict.

    Expected config keys (from your depth_unet_cfg):
      - model_type (ignored, for readability)
      - addition_time_embed_dim, block_out_channels, cross_attention_dim, ...
      - in_channels, out_channels, num_frames, sample_size, ...
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        # prepare kwargs: remove non-constructor keys and convert lists->tuples
        allowed = {
            "sample_size", "in_channels", "out_channels",
            "down_block_types", "up_block_types",
            "block_out_channels", "addition_time_embed_dim",
            "projection_class_embeddings_input_dim",
            "layers_per_block", "cross_attention_dim",
            "transformer_layers_per_block", "num_attention_heads",
            "num_frames"
        }
        unet_kwargs = {}
        for k, v in cfg.items():
            if k in allowed:
                unet_kwargs[k] = tuple(v) if isinstance(v, list) else v

        self.unet_config = copy.deepcopy(unet_kwargs)

        # instantiate UNet
        self.unet = UNetSpatioTemporalConditionModel(**unet_kwargs)
        self.config = getattr(self.unet, "config", None)

        # prepare a projector for encoder_hidden_states if its dim != cross_attention_dim
        self.cross_attention_dim = unet_kwargs.get("cross_attention_dim", None)
        # We don't know encoder_hidden_states incoming dim; create projector lazily on first forward if needed
        self._cond_projector: Optional[nn.Module] = None

        self.device = "cuda"

    def _find_time_embedding_module(self):
        """
        Try to locate an nn.Embedding used for added_time_ids.
        Return (name, module) or (None, None).
        """
        for name, module in self.unet.named_modules():
            if "time" in name.lower() and isinstance(module, nn.Embedding):
                return name, module
        return None, None

    def _prepare_added_time_ids(self, batch_size: int, num_frames: int, device: torch.device):
        return torch.ones((batch_size,), device=self.device) * self.unet_config["num_frames"]


    def _ensure_cond_projection(self, encoder_hidden_states: torch.Tensor):
        """
        If encoder_hidden_states last dim != cross_attention_dim, create a Linear projection layer
        to map it -> cross_attention_dim. The projector is created once and reused.
        """
        if encoder_hidden_states is None:
            return None
        if self.cross_attention_dim is None:
            return encoder_hidden_states  # nothing to do

        in_dim = encoder_hidden_states.shape[-1]
        if in_dim == self.cross_attention_dim:
            return encoder_hidden_states
        # lazy create projector
        if self._cond_projector is None:
            self._cond_projector = nn.Linear(in_dim, self.cross_attention_dim).to(self.device)
        # project
        B, L, _ = encoder_hidden_states.shape
        flat = encoder_hidden_states.view(-1, in_dim)
        proj = self._cond_projector(flat).view(B, L, self.cross_attention_dim)
        return proj

    def forward(self,
                sample: torch.Tensor,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                return_dict: bool = True):
        """
        sample: (B, T, C_in, H, W)  -- must match unet.in_channels and num_frames
        encoder_hidden_states: optional (B, L, D_in) condition sequence
        """
        device = self.device
        sample = sample.to(device)

        B = sample.shape[0]
        # enforce timestep = 0 (per your request), model accepts scalar/ tensor
        timestep = torch.zeros(B, dtype=torch.float32, device=device)

        # prepare added_time_ids as 0,0.1,...0.9 for T frames (repeat over batch)
        num_frames = sample.shape[1]
        added_time_ids = self._prepare_added_time_ids(B, num_frames, device)

        # project encoder_hidden_states if necessary
        if encoder_hidden_states is not None:
            # ensure float tensor on device
            encoder_hidden_states = encoder_hidden_states.to(device)
            # if encoder_hidden_states is shape (B, D) -> convert to (B,1,D)
            if encoder_hidden_states.dim() == 2:
                encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
            encoder_hidden_states = self._ensure_cond_projection(encoder_hidden_states)

        # call UNet

        # print("sample", sample.shape)

        out = self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_time_ids=added_time_ids,
            return_dict=return_dict
        )

        # The UNetSpatioTemporalConditionModel usually returns a dataclass with .sample attribute.
        if return_dict and hasattr(out, "sample"):
            return out
        else:
            # If model returns raw tensor, just return it
            return out


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="7"

    depth_unet_cfg = {
        "model_type": "UNetSpatioTemporalConditionModel",
        "addition_time_embed_dim": 768,
        "block_out_channels": [320, 320, 640, 640],
        "cross_attention_dim": 384,
        "down_block_types": [
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "DownBlockSpatioTemporal",
        ],
        "in_channels": 4,
        "layers_per_block": 2,
        "num_attention_heads": [5, 10, 20, 20],
        "num_frames": 10,
        "out_channels": 4,
        "projection_class_embeddings_input_dim": 768,
        "sample_size": 32,
        "transformer_layers_per_block": 1,
        "up_block_types": [
            "UpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
        ],
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Depth_UNet(depth_unet_cfg).cuda()

    # dummy inputs
    B = 2
    T = depth_unet_cfg["num_frames"]
    C_in = depth_unet_cfg["in_channels"]
    H = W = depth_unet_cfg["sample_size"]
    sample = torch.randn(B, T, C_in, H, W, device=device)
    print("Input sample shape:", sample.shape)


    # dummy condition: (B, L, D_cond) where D_cond can be any; projector will adapt if needed
    cond = torch.randn(B, 336, 384, device=device)  # e.g., 5 tokens of dim 512

    out = model(sample=sample, encoder_hidden_states=cond, return_dict=True)
    # 如果 return_dict 且返回对象有 .sample，打印 shape
    if hasattr(out, "sample"):
        print("Output sample shape:", out.sample.shape)
    else:
        print("Output tensor shape:", out.shape)
