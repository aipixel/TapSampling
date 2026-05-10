from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution, Normal

from action_vae.model.transformer import (
    SequencePositionalEncoding,
    SkipTransformerDecoder,
    SkipTransformerEncoder,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)


class AutoMldVae(nn.Module):
    def __init__(
        self,
        nfeats: int,
        latent_dim: tuple[int, int] = (1, 256),
        h_dim: int = 512,
        ff_size: int = 1024,
        num_layers: int = 9,
        num_heads: int = 4,
        dropout: float = 0.1,
        arch: str = "all_encoder",
        normalize_before: bool = False,
        activation: str = "gelu",
        position_embedding: str = "learned",
        **_: object,
    ) -> None:
        super().__init__()

        self.latent_size = latent_dim[0]
        self.latent_dim = latent_dim[-1]
        self.h_dim = h_dim
        self.arch = arch

        self.query_pos_encoder = SequencePositionalEncoding(self.h_dim, position_embedding=position_embedding)
        self.query_pos_decoder = SequencePositionalEncoding(self.h_dim, position_embedding=position_embedding)

        encoder_layer = TransformerEncoderLayer(
            self.h_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )
        encoder_norm = nn.LayerNorm(self.h_dim)
        self.encoder = SkipTransformerEncoder(encoder_layer, num_layers, encoder_norm)
        self.encoder_latent_proj = nn.Linear(self.h_dim, self.latent_dim)

        if self.arch == "all_encoder":
            decoder_norm = nn.LayerNorm(self.h_dim)
            self.decoder = SkipTransformerEncoder(encoder_layer, num_layers, decoder_norm)
        elif self.arch == "encoder_decoder":
            decoder_layer = TransformerDecoderLayer(
                self.h_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
            decoder_norm = nn.LayerNorm(self.h_dim)
            self.decoder = SkipTransformerDecoder(decoder_layer, num_layers, decoder_norm)
        else:
            raise ValueError("Not support architecture!")

        self.decoder_latent_proj = nn.Linear(self.latent_dim, self.h_dim)
        self.global_motion_token = nn.Parameter(torch.randn(self.latent_size * 2, self.h_dim))
        self.skel_embedding = nn.Linear(nfeats, self.h_dim)
        self.final_layer = nn.Linear(self.h_dim, nfeats)

        self.register_buffer("latent_mean", torch.tensor(0.0))
        self.register_buffer("latent_std", torch.tensor(1.0))

    def encode(
        self,
        future_motion: Tensor,
        history_motion: Tensor,
        scale_latent: bool = False,
    ) -> tuple[Tensor, Distribution]:
        batch_size = future_motion.shape[0]
        motion = torch.cat((history_motion, future_motion), dim=1)
        motion = self.skel_embedding(motion).permute(1, 0, 2)

        dist_tokens = torch.tile(self.global_motion_token[:, None, :], (1, batch_size, 1))
        encoded = torch.cat((dist_tokens, motion), dim=0)
        encoded = self.query_pos_encoder(encoded)
        encoded = self.encoder(encoded)[: dist_tokens.shape[0]]
        encoded = self.encoder_latent_proj(encoded)

        mu = encoded[: self.latent_size]
        logvar = encoded[self.latent_size :]
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std = logvar.exp().pow(0.5)
        dist = Normal(mu, std)
        latent = dist.rsample()
        if scale_latent:
            latent = latent / self.latent_std
        return latent, dist

    def decode(
        self,
        z: Tensor,
        history_motion: Tensor,
        nfuture: int,
        scale_latent: bool = False,
    ) -> Tensor:
        batch_size = history_motion.shape[0]
        if scale_latent:
            z = z * self.latent_std

        z = self.decoder_latent_proj(z)
        queries = torch.zeros(nfuture, batch_size, self.h_dim, device=z.device)
        history_embedding = self.skel_embedding(history_motion).permute(1, 0, 2)

        if self.arch == "all_encoder":
            decoded = torch.cat((z, history_embedding, queries), dim=0)
            decoded = self.query_pos_decoder(decoded)
            decoded = self.decoder(decoded)[-nfuture:]
        else:
            decoded = torch.cat((history_embedding, queries), dim=0)
            decoded = self.query_pos_decoder(decoded)
            decoded = self.decoder(tgt=decoded, memory=z)[-nfuture:]

        decoded = self.final_layer(decoded)
        return decoded.permute(1, 0, 2)
