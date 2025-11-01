"""Manifold diffusion model for high-dimensional time-series imputation.

This module implements a prototype pipeline that maps multivariate time-series
onto a hyperspherical manifold, trains a denoising diffusion probabilistic
model (DDPM) in the tangent space of a reference point, and performs
conditional sampling to fill in missing values. The implementation focuses on
clarity and extensibility rather than maximum efficiency.

Requirements
------------
- PyTorch >= 2.0
- Geoopt >= 0.5 (for manifold-aware operations, optional but recommended)

Usage outline
-------------
```
config = DiffusionConfig(
    timesteps=1000,
    beta_start=1e-4,
    beta_end=0.02,
    latent_dim=32,
    hidden_dim=128,
)

model = ManifoldDiffusionImputer(input_dim=feature_dim, config=config)

for batch in dataloader:
    loss = model.training_step(batch["x"], batch["mask"])
    loss.backward()
    optimizer.step()

with torch.no_grad():
    imputations = model.impute(x_with_gaps, mask)
```

The code is organized so that practitioners can swap in alternative geometric
structures (e.g. hyperbolic manifolds, learned manifolds) by replacing the
`SphereManifold` helper with another adapter that exposes `expmap`, `logmap`,
`project`, and `to_tangent` methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DiffusionConfig:
    """Hyperparameters controlling the diffusion process and model size."""

    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule: str = "cosine"
    latent_dim: int = 32
    hidden_dim: int = 256
    dropout: float = 0.1
    use_geoopt: bool = False  # Toggle to True if geoopt is available.


def _linear_beta_schedule(timesteps: int, start: float, end: float) -> torch.Tensor:
    return torch.linspace(start, end, timesteps)


def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(min=1e-5, max=0.999)


def build_beta_schedule(config: DiffusionConfig) -> torch.Tensor:
    if config.schedule == "linear":
        return _linear_beta_schedule(config.timesteps, config.beta_start, config.beta_end)
    if config.schedule == "cosine":
        return _cosine_beta_schedule(config.timesteps)
    raise ValueError(f"Unsupported beta schedule: {config.schedule}")


class SphereManifold:
    """Utility class for computations on a unit hypersphere S^{d-1}."""

    def __init__(self, dim: int, eps: float = 1e-6):
        self.dim = dim
        self.eps = eps
        origin = torch.zeros(dim)
        origin[0] = 1.0
        self.register_origin(origin)

    def register_origin(self, origin: torch.Tensor) -> None:
        self._origin = origin

    @property
    def origin(self) -> torch.Tensor:
        return self._origin

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1, eps=self.eps)

    def to_tangent(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Remove radial component: ensure v is orthogonal to x.
        inner = torch.sum(x * v, dim=-1, keepdim=True)
        return v - inner * x

    def expmap(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        norm_v = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(self.eps)
        direction = v / norm_v
        cos_term = torch.cos(norm_v)
        sin_term = torch.sin(norm_v)
        y = cos_term * x + sin_term * direction
        return self.project(y)

    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        inner = (x * y).sum(dim=-1, keepdim=True).clamp(-1.0 + self.eps, 1.0 - self.eps)
        theta = torch.acos(inner)
        sin_theta = torch.sin(theta).clamp_min(self.eps)
        direction = self.to_tangent(x, y - inner * x)
        direction = direction / sin_theta
        return direction * theta

    def geodesic_interpolate(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        v = self.logmap(x, y)
        return self.expmap(x, v * t.unsqueeze(-1))

    def broadcast_origin(self, batch_shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        origin = self.origin.to(device)
        expand_shape = batch_shape + origin.shape
        return origin.view(*((1,) * len(batch_shape)), -1).expand(expand_shape)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        freqs = torch.exp(
            torch.arange(half_dim, device=t.device, dtype=t.dtype)
            * -(torch.log(torch.tensor(10000.0, device=t.device, dtype=t.dtype)) / (half_dim - 1))
        )
        args = t.float().unsqueeze(-1) * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm1(x))


class ScoreNetwork(nn.Module):
    """Predicts noise in tangent coordinates given latent states and masks."""

    def __init__(self, latent_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(latent_dim)
        self.input_proj = nn.Linear(latent_dim * 2 + 1, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim * 4, dropout) for _ in range(4)
        ])
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, u_t: torch.Tensor, t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, seq_len, latent_dim = u_t.shape
        t_embed = self.time_embed(t).unsqueeze(1).expand(b, seq_len, -1)
        inputs = torch.cat([u_t, t_embed, mask], dim=-1)
        h = self.input_proj(inputs)
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)


class TemporalEncoder(nn.Module):
    """GRU-based encoder that maps x -> hyperspherical latent representation."""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        hidden_dim = latent_dim * 2
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        return self.proj(h)


class TemporalDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        hidden_dim = latent_dim * 2
        self.gru = nn.GRU(input_size=latent_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(z)
        return self.proj(h)


class DiffusionScheduler:
    def __init__(self, config: DiffusionConfig):
        betas = build_beta_schedule(config)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register(betas, alphas, alphas_cumprod)

    def register(self, betas: torch.Tensor, alphas: torch.Tensor, alphas_cumprod: torch.Tensor) -> None:
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=alphas.device), alphas_cumprod[:-1]])
        self.sqrt_alphas = torch.sqrt(alphas)
        self.sqrt_inv_alphas = 1.0 / self.sqrt_alphas
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        ).clamp(min=1e-9)

    def to(self, device: torch.device) -> "DiffusionScheduler":
        for attr in [
            "betas",
            "alphas",
            "alphas_cumprod",
            "alphas_cumprod_prev",
            "sqrt_alphas",
            "sqrt_inv_alphas",
            "sqrt_one_minus_alphas_cumprod",
            "posterior_variance",
        ]:
            tensor = getattr(self, attr)
            setattr(self, attr, tensor.to(device))
        return self


class ManifoldDiffusionImputer(nn.Module):
    """High-level wrapper that orchestrates encoding, diffusion, and decoding."""

    def __init__(self, input_dim: int, config: Optional[DiffusionConfig] = None):
        super().__init__()
        self.config = config or DiffusionConfig()
        self.scheduler = DiffusionScheduler(self.config)
        self.manifold = SphereManifold(self.config.latent_dim)

        self.encoder = TemporalEncoder(input_dim=input_dim, latent_dim=self.config.latent_dim)
        self.decoder = TemporalDecoder(latent_dim=self.config.latent_dim, output_dim=input_dim)
        self.score_model = ScoreNetwork(
            latent_dim=self.config.latent_dim,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
        )

    def _origin(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        return self.manifold.broadcast_origin(shape, device)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.manifold.project(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def _logmap_origin(self, z: torch.Tensor) -> torch.Tensor:
        origin = self._origin(z.shape[:-1], z.device)
        return self.manifold.logmap(origin, z)

    def _expmap_origin(self, u: torch.Tensor) -> torch.Tensor:
        origin = self._origin(u.shape[:-1], u.device)
        return self.manifold.expmap(origin, u)

    def _q_sample(self, u0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # u0, noise shape: (B, T, D)
        alphas_cumprod = self.scheduler.alphas_cumprod.to(u0.device)
        sqrt_alpha = alphas_cumprod[t].view(-1, 1, 1).sqrt()
        sqrt_one_minus = self.scheduler.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_alpha * u0 + sqrt_one_minus * noise

    def _prepare_noise(self, z: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(z)
        return self.manifold.to_tangent(z, noise)

    def training_step(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute diffusion loss for one batch.

        Args:
            x: [B, L, D] input time-series with missing entries (filled with zeros or stats).
            mask: same shape, binary indicator where 1 marks observed values.
        """

        device = x.device
        b, seq_len, _ = x.shape
        self.scheduler.to(device)

        z = self.encode(x)
        u0 = self._logmap_origin(z)

        t = torch.randint(0, self.config.timesteps, (b,), device=device)
        noise = torch.randn_like(u0)
        u_t = self._q_sample(u0, noise, t)

        pred_noise = self.score_model(u_t, t, mask)
        loss = F.mse_loss(pred_noise, noise, reduction="none")
        # Focus loss on missing positions (mask == 0) while keeping stability.
        missing_weight = (1.0 - mask).clamp_min(0.1)
        loss = (loss * missing_weight).mean()
        return loss

    @torch.no_grad()
    def p_sample(self, u_t: torch.Tensor, t: int, mask: torch.Tensor, u_observed: torch.Tensor) -> torch.Tensor:
        beta_t = self.scheduler.betas[t]
        sqrt_one_minus = self.scheduler.sqrt_one_minus_alphas_cumprod[t]
        sqrt_inv_alpha = self.scheduler.sqrt_inv_alphas[t]
        posterior_var = self.scheduler.posterior_variance[t]

        t_tensor = torch.full((u_t.size(0),), t, device=u_t.device, dtype=torch.long)
        eps_theta = self.score_model(u_t, t_tensor, mask)

        model_mean = sqrt_inv_alpha * (u_t - beta_t / sqrt_one_minus * eps_theta)
        if t > 0:
            noise = torch.randn_like(u_t)
            sample = model_mean + torch.sqrt(posterior_var) * noise
        else:
            sample = model_mean

        # Re-impose observed latents to keep consistency.
        return mask * u_observed + (1.0 - mask) * sample

    @torch.no_grad()
    def impute(self, x: torch.Tensor, mask: torch.Tensor, num_steps: Optional[int] = None) -> torch.Tensor:
        """Run conditional reverse diffusion to impute missing values.

        Args:
            x: [B, L, D] tensor with placeholders for missing entries.
            mask: same shape, 1 for observed entries, 0 for missing.
            num_steps: override number of reverse diffusion steps.

        Returns:
            Imputed series of shape [B, L, D].
        """

        device = x.device
        steps = num_steps or self.config.timesteps
        if steps > self.config.timesteps:
            raise ValueError("num_steps cannot exceed configured timesteps")

        self.scheduler.to(device)
        z_obs = self.encode(x)
        u_observed = self._logmap_origin(z_obs)

        u = torch.randn_like(u_observed)

        for step in reversed(range(steps)):
            u = self.p_sample(u, step, mask, u_observed)

        z0 = self._expmap_origin(u)
        return self.decode(z0)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target) ** 2
    diff = diff * (1.0 - mask)
    normalization = (1.0 - mask).sum().clamp_min(1.0)
    return diff.sum() / normalization


__all__ = [
    "DiffusionConfig",
    "DiffusionScheduler",
    "ManifoldDiffusionImputer",
    "masked_mse",
]
