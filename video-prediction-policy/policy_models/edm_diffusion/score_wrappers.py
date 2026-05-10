from torch import nn
import torch
from .utils import append_dims
from policy_models.module.diffusion_decoder import DiffusionTransformer, Dual_DiffusionTransformer, Dual_DiffusionTransformer_Dual_Encoder

'''
Wrappers for the score-based models based on Karras et al. 2022
They are used to get improved scaling of different noise levels, which
improves training stability and model performance 

Code is adapted from:

https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/layers.py
'''


class GCDenoiser(nn.Module):
    """
    A Karras et al. preconditioner for denoising diffusion models.

    Args:
        inner_model: The inner model used for denoising.
        sigma_data: The data sigma for scalings (default: 1.0).
    """
    def __init__(self, action_dim, obs_dim, goal_dim, num_tokens, goal_window_size, obs_seq_len, act_seq_len, device, sigma_data=1., proprio_dim=8,
                 dual_former = False,
                 skeleton_dim: int = 21,
                 bidirection: bool = True,
                 cross_after_blocks = False,
                 gcdenoiser_decoder_layer = 4, 
                 gcdenoiser_dual_adapter = False,
                 skeleton_detach = False,

                 use_Dual_DiffusionTransformer_Dual_Encoder = False
    ):
        super().__init__()
        self.dual_former = dual_former


        if use_Dual_DiffusionTransformer_Dual_Encoder:
            print("use Dual_DiffusionTransformer_Dual_Encoder, bridge attention")
            self.inner_model = Dual_DiffusionTransformer_Dual_Encoder(
                    action_dim = action_dim,
                    obs_dim = obs_dim,
                    goal_dim = goal_dim,
                    proprio_dim= proprio_dim,
                    goal_conditioned = True,
                    embed_dim = 384,
                    n_dec_layers = gcdenoiser_decoder_layer,
                    n_enc_layers = 4,
                    n_obs_token = num_tokens,
                    goal_seq_len = goal_window_size,
                    obs_seq_len = obs_seq_len,
                    action_seq_len =act_seq_len,
                    embed_pdrob = 0,
                    goal_drop = 0,
                    attn_pdrop = 0.3,
                    resid_pdrop = 0.1,
                    mlp_pdrop = 0.05,
                    n_heads= 8,
                    device = device,
                    use_mlp_goal = True,

                    skeleton_dim = skeleton_dim,
                    bidirection = bidirection,
                    cross_after_blocks = cross_after_blocks,
                    gcdenoiser_dual_adapter = gcdenoiser_dual_adapter,
                    skeleton_detach = skeleton_detach
            )
        else:
            if not dual_former:
                self.inner_model = DiffusionTransformer(
                    action_dim = action_dim,
                    obs_dim = obs_dim,
                    goal_dim = goal_dim,
                    proprio_dim= proprio_dim,
                    goal_conditioned = True,
                    embed_dim = 384,
                    n_dec_layers = 4,
                    n_enc_layers = 4,
                    n_obs_token = num_tokens,
                    goal_seq_len = goal_window_size,
                    obs_seq_len = obs_seq_len,
                    action_seq_len =act_seq_len,
                    embed_pdrob = 0,
                    goal_drop = 0,
                    attn_pdrop = 0.3,
                    resid_pdrop = 0.1,
                    mlp_pdrop = 0.05,
                    n_heads= 8,
                    device = device,
                    use_mlp_goal = True,
                )
            else:
                self.inner_model = Dual_DiffusionTransformer(
                    action_dim = action_dim,
                    obs_dim = obs_dim,
                    goal_dim = goal_dim,
                    proprio_dim= proprio_dim,
                    goal_conditioned = True,
                    embed_dim = 384,
                    n_dec_layers = gcdenoiser_decoder_layer,
                    n_enc_layers = 4,
                    n_obs_token = num_tokens,
                    goal_seq_len = goal_window_size,
                    obs_seq_len = obs_seq_len,
                    action_seq_len =act_seq_len,
                    embed_pdrob = 0,
                    goal_drop = 0,
                    attn_pdrop = 0.3,
                    resid_pdrop = 0.1,
                    mlp_pdrop = 0.05,
                    n_heads= 8,
                    device = device,
                    use_mlp_goal = True,

                    skeleton_dim = skeleton_dim,
                    bidirection = bidirection,
                    cross_after_blocks = cross_after_blocks,
                    gcdenoiser_dual_adapter = gcdenoiser_dual_adapter,
                    skeleton_detach = skeleton_detach
                )
        self.sigma_data = sigma_data

    def get_scalings(self, sigma):
        """
        Compute the scalings for the denoising process.

        Args:
            sigma: The input sigma.
        Returns:
            The computed scalings for skip connections, output, and input.
        """
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def loss(self, state, action, goal, noise, sigma, **kwargs):
        """
        Compute the loss for the denoising process.

        Args:
            state: The input state.
            action: The input action.
            goal: The input goal.
            noise: The input noise.
            sigma: The input sigma.
            **kwargs: Additional keyword arguments.
        Returns:
            The computed loss.
        """
        c_skip, c_out, c_in = [append_dims(x, action.ndim) for x in self.get_scalings(sigma)]
        noised_input = action + noise * append_dims(sigma, action.ndim)
        model_output = self.inner_model(state, noised_input * c_in, goal, sigma, **kwargs)

        if not self.dual_former:
            target = (action - c_skip * noised_input) / c_out
            return (model_output - target).pow(2).flatten(1).mean(), model_output
        else:
            target_action, target_skeleton = torch.split((action - c_skip * noised_input) / c_out, [7, action.shape[-1] - 7], dim = -1)
            model_output_action, model_output_skeleton = torch.split(model_output, [7, action.shape[-1] - 7], dim = -1)

            return (model_output_action - target_action).pow(2).flatten(1).mean(), \
                   (model_output_skeleton - target_skeleton).pow(2).flatten(1).mean(), \
                   model_output

    def forward(self, state, action, goal, sigma, **kwargs):
        """
        Perform the forward pass of the denoising process.

        Args:
            state: The input state.
            action: The input action.
            goal: The input goal.
            sigma: The input sigma.
            **kwargs: Additional keyword arguments.

        Returns:
            The output of the forward pass.
        """
        c_skip, c_out, c_in = [append_dims(x, action.ndim) for x in self.get_scalings(sigma)]
        return self.inner_model(state, action * c_in, goal, sigma, **kwargs) * c_out + action * c_skip
    
    def forward_context_only(self, state, action, goal, sigma, **kwargs):
        """
        Perform the forward pass of the denoising process.

        Args:
            state: The input state.
            action: The input action.
            goal: The input goal.
            sigma: The input sigma.
            **kwargs: Additional keyword arguments.

        Returns:
            The output of the forward pass.
        """
        c_skip, c_out, c_in = [append_dims(x, action.ndim) for x in self.get_scalings(sigma)]
        return self.inner_model.forward_enc_only(state, action * c_in, goal, sigma, **kwargs)

    def get_params(self):
        return self.inner_model.parameters()
