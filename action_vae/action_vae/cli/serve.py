from __future__ import annotations

import argparse
from typing import List

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from action_vae.common.config import ServeConfig
from action_vae.common.inference import ActionVAEInference


class GenerateRequest(BaseModel):
    actions: List[List[List[float]]]
    M: int
    replacement: bool = True
    use_rsample: bool = False


class GenerateResponse(BaseModel):
    generated: List[List[List[float]]]


def parse_args() -> ServeConfig:
    defaults = ServeConfig()
    parser = argparse.ArgumentParser(
        description="Serve the standalone action VAE over FastAPI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--window_size", type=int, default=defaults.window_size)
    parser.add_argument("--feature_dim", type=int, default=defaults.feature_dim)
    parser.add_argument("--num_layers", type=int, default=defaults.num_layers)
    parser.add_argument("--latent_tokens", type=int, default=defaults.latent_tokens)
    parser.add_argument("--latent_dim", type=int, default=defaults.latent_dim)
    parser.add_argument("--host", default=defaults.host)
    parser.add_argument("--port", type=int, default=defaults.port)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default=defaults.device)
    namespace = parser.parse_args()
    return ServeConfig(
        checkpoint_path=namespace.checkpoint_path,
        window_size=namespace.window_size,
        feature_dim=namespace.feature_dim,
        num_layers=namespace.num_layers,
        latent_tokens=namespace.latent_tokens,
        latent_dim=namespace.latent_dim,
        host=namespace.host,
        port=namespace.port,
        seed=namespace.seed,
        device=namespace.device,
    )


def build_app(service: ActionVAEInference):
    app = FastAPI(title="Action Generation Service")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/generate", response_model=GenerateResponse)
    def generate_actions(req: GenerateRequest) -> GenerateResponse:
        try:
            actions = np.asarray(req.actions, dtype=np.float32)
            print("input actions:", actions.shape)
            print("target sample num:", req.M)

            if actions.ndim != 3 or actions.shape[1] != service.window_size or actions.shape[2] != service.feature_dim:
                raise ValueError(
                    f"Expected shape (N,{service.window_size},{service.feature_dim}), got {tuple(actions.shape)}"
                )

            _, sampled_latents = service.encode_actions_into_dist(
                actions,
                sample_actions=req.M,
                replacement=req.replacement,
                use_rsample=req.use_rsample,
            )
            if sampled_latents is None:
                raise RuntimeError("Failed to sample latents")

            generated = service.decode_latents_into_actions(sampled_latents)
            if torch.is_tensor(generated):
                generated = generated.detach().cpu().numpy()

            print("sample actions:", generated.shape)
            return GenerateResponse(generated=generated.tolist())
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)

    args = parse_args()
    service = ActionVAEInference(args)
    app = build_app(service)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
