"""Conditional diffusion model for time-series representation learning.

This module implements a compact conditional diffusion framework tailored for
time-series data. It provides:

* A synthetic sinusoid dataset with covariates used as conditioning signals.
* A lightweight 1D residual network that predicts diffusion noise conditioned
  on auxiliary variables.
* A Gaussian diffusion wrapper that exposes training, sampling, and
  representation extraction utilities.
* A simple training loop that showcases how to learn representations from
  time-series via conditional diffusion.

Example usage (from shell):

```
python src/conditional_diffusion_timeseries.py --epochs 5 --device auto
```
"""

from __future__ import annotations

import argparse
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """Create sinusoidal timestep embeddings.

    Args:
        timesteps: Integer tensor of shape [batch] with diffusion steps.
        embedding_dim: Size of the embedding vector.

    Returns:
        Tensor of shape [batch, embedding_dim].
    """

    half_dim = embedding_dim // 2
    device = timesteps.device
    exponent = torch.arange(half_dim, device=device, dtype=torch.float32)
    exponent = -math.log(10000.0) * exponent / max(half_dim - 1, 1)
    freqs = torch.exp(exponent)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TimeEmbedding(nn.Module):
    """Two-layer MLP on top of sinusoidal timestep embeddings."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        hidden = embedding_dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, embedding_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.mlp(get_timestep_embedding(timesteps, self.embedding_dim))


class ResidualBlock(nn.Module):
    """Residual block with time and condition modulation."""

    def __init__(
        self,
        channels: int,
        time_embed_dim: int,
        cond_embed_dim: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.time_proj = nn.Linear(time_embed_dim, channels)
        self.cond_proj = nn.Linear(cond_embed_dim, channels)

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
        cond_embedding: torch.Tensor,
    ) -> torch.Tensor:
        residual = x
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(time_embedding).unsqueeze(-1)
        h = h + self.cond_proj(cond_embedding).unsqueeze(-1)
        h = self.conv2(self.act(self.norm2(h)))
        return h + residual


class ConditionalDiffusionUNet(nn.Module):
    """Lightweight conditional diffusion backbone for 1D signals."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        num_layers: int,
        cond_dim: int,
        time_embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.time_embed = TimeEmbedding(time_embed_dim)
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.input_proj = nn.Conv1d(input_channels, hidden_channels, kernel_size=1)
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, input_channels, kernel_size=1),
        )

        dilations = [min(2 ** i, 16) for i in range(num_layers)]
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    channels=hidden_channels,
                    time_embed_dim=time_embed_dim,
                    cond_embed_dim=time_embed_dim,
                    dilation=dilation,
                )
                for dilation in dilations
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        cond: torch.Tensor,
        return_latent: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        t_embed = self.time_embed(timesteps)
        c_embed = self.cond_embed(cond)

        h = self.input_proj(x)
        features = []
        for block in self.blocks:
            h = block(h, t_embed, c_embed)
            features.append(h)

        out = self.output_proj(h)

        if return_latent:
            latent = torch.cat([feat.mean(dim=-1) for feat in features], dim=-1)
            return out, latent
        return out


class ConditionalDiffusionModel(nn.Module):
    """Wrapper providing noise prediction and representation extraction."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        num_layers: int,
        cond_dim: int,
        time_embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.network = ConditionalDiffusionUNet(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            cond_dim=cond_dim,
            time_embed_dim=time_embed_dim,
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.network(x, timesteps, cond, return_latent=False)

    @torch.no_grad()
    def get_representation(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        timesteps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if timesteps is None:
            timesteps = torch.zeros(x.size(0), device=x.device, dtype=torch.long)
        _, latent = self.network(x, timesteps, cond, return_latent=True)
        return latent


def make_beta_schedule(schedule: str, timesteps: int) -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(1e-4, 0.02, timesteps)
    if schedule == "cosine":
        steps = torch.arange(timesteps + 1, dtype=torch.float32)
        s = 0.008
        alphas_cumprod = torch.cos(((steps / timesteps + s) / (1 + s)) * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(0, 0.999)
    raise ValueError(f"Unknown beta schedule: {schedule}")


class GaussianDiffusion(nn.Module):
    """Gaussian diffusion process with conditional noise predictor."""

    def __init__(
        self,
        model: ConditionalDiffusionModel,
        timesteps: int = 1000,
        beta_schedule: str = "cosine",
    ) -> None:
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        betas = make_beta_schedule(beta_schedule, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("posterior_variance", betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alphas = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_alphas * x_start + sqrt_one_minus * noise

    def p_mean_variance(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        guidance_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if guidance_scale != 0.0:
            cond_null = torch.zeros_like(cond)
            eps_cond = self.model(x_t, t, cond)
            eps_uncond = self.model(x_t, t, cond_null)
            eps_theta = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            eps_theta = self.model(x_t, t, cond)

        beta_t = self.betas[t].view(-1, 1, 1)
        sqrt_recip_alpha = self.sqrt_recip_alphas[t].view(-1, 1, 1)
        sqrt_one_minus_cumprod = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)

        mean = sqrt_recip_alpha * (x_t - beta_t / sqrt_one_minus_cumprod * eps_theta)
        var = self.posterior_variance[t].view(-1, 1, 1)
        return mean, var

    def training_losses(self, x_start: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        batch_size = x_start.size(0)
        device = x_start.device
        t = torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        noise_pred = self.model(x_noisy, t, cond)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        sample_shape: Tuple[int, int, int],
        guidance_scale: float = 0.0,
    ) -> torch.Tensor:
        device = cond.device
        x = torch.randn(sample_shape, device=device)
        for step in reversed(range(self.timesteps)):
            t = torch.full((cond.size(0),), step, device=device, dtype=torch.long)
            mean, var = self.p_mean_variance(x, t, cond, guidance_scale=guidance_scale)
            if step > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(var) * noise
            else:
                x = mean
        return x

    @torch.no_grad()
    def encode(self, x: torch.Tensor, cond: torch.Tensor, timesteps: torch.Tensor | None = None) -> torch.Tensor:
        return self.model.get_representation(x, cond, timesteps)


class SinusoidDataset(Dataset):
    """Synthetic conditioned time-series dataset for representation learning."""

    def __init__(
        self,
        num_samples: int = 10000,
        sequence_length: int = 128,
        noise_std: float = 0.05,
        seed: int = 13,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.noise_std = noise_std
        rng = torch.Generator().manual_seed(seed)
        self.time_index = torch.linspace(0, 1, sequence_length)

        amplitude = torch.empty(num_samples).uniform_(0.3, 1.3, generator=rng)
        frequency = torch.empty(num_samples).uniform_(0.5, 2.5, generator=rng)
        phase = torch.empty(num_samples).uniform_(0.0, math.pi, generator=rng)
        trend = torch.empty(num_samples).uniform_(-0.3, 0.3, generator=rng)

        self.conditions = torch.stack([amplitude, frequency, phase, trend], dim=-1)

    def __len__(self) -> int:
        return self.conditions.shape[0]

    def _generate_signal(self, cond: torch.Tensor) -> torch.Tensor:
        amplitude, frequency, phase, trend = cond
        base = amplitude * torch.sin(2 * math.pi * frequency * self.time_index + phase)
        base = base + trend * self.time_index
        return base

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cond = self.conditions[index]
        clean = self._generate_signal(cond)
        noisy = clean + torch.randn_like(clean) * self.noise_std
        return noisy.unsqueeze(0), cond

    def sample_conditions(self, batch_size: int, device: torch.device) -> torch.Tensor:
        idx = torch.randint(0, len(self), (batch_size,))
        return self.conditions[idx].to(device)


def train(args: argparse.Namespace) -> None:
    device = (
        torch.device(args.device)
        if args.device != "auto"
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    dataset = SinusoidDataset(
        num_samples=args.num_samples,
        sequence_length=args.sequence_length,
        noise_std=args.noise_std,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = ConditionalDiffusionModel(
        input_channels=1,
        hidden_channels=args.hidden_channels,
        num_layers=args.layers,
        cond_dim=dataset.conditions.shape[-1],
        time_embed_dim=args.time_embed_dim,
    )
    diffusion = GaussianDiffusion(
        model=model,
        timesteps=args.timesteps,
        beta_schedule=args.beta_schedule,
    ).to(device)

    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=args.lr)

    global_step = 0
    diffusion.train()
    for epoch in range(args.epochs):
        for x, cond in dataloader:
            x = x.to(device)
            cond = cond.to(device)

            loss = diffusion.training_losses(x, cond)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(diffusion.parameters(), args.grad_clip)
            optimizer.step()

            if global_step % args.log_interval == 0:
                print(
                    f"epoch={epoch} step={global_step} loss={loss.item():.4f}"
                )

            if global_step % args.eval_interval == 0:
                diffusion.eval()
                with torch.no_grad():
                    eval_x = x[: args.eval_samples]
                    eval_cond = cond[: args.eval_samples]
                    latent = diffusion.encode(eval_x, eval_cond)
                    latent_mean = latent.mean().item()
                    latent_std = latent.std(unbiased=False).item()
                    samples = diffusion.sample(
                        eval_cond,
                        sample_shape=(eval_cond.size(0), 1, args.sequence_length),
                        guidance_scale=args.guidance_scale,
                    )
                    reconstruction_error = F.mse_loss(samples, eval_x, reduction="mean").item()
                    print(
                        f"eval latent_mean={latent_mean:.4f} latent_std={latent_std:.4f} "
                        f"sample_mse={reconstruction_error:.4f}"
                    )
                diffusion.train()

            global_step += 1

    diffusion.eval()
    with torch.no_grad():
        cond = dataset.sample_conditions(args.eval_samples, device)
        samples = diffusion.sample(
            cond,
            sample_shape=(cond.size(0), 1, args.sequence_length),
            guidance_scale=args.guidance_scale,
        )
        latent = diffusion.encode(samples, cond)
        print(
            "Training complete. Representation tensor shape:",
            tuple(latent.shape),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a conditional diffusion model for time-series representations.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for AdamW.")
    parser.add_argument(
        "--timesteps", type=int, default=400, help="Number of diffusion steps."
    )
    parser.add_argument(
        "--beta-schedule",
        type=str,
        default="cosine",
        choices=["cosine", "linear"],
        help="Noise schedule for betas.",
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=128,
        help="Hidden channel width of the denoiser network.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=6,
        help="Number of residual layers in the denoiser.",
    )
    parser.add_argument(
        "--time-embed-dim",
        type=int,
        default=128,
        help="Dimensionality of time and condition embeddings.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=128,
        help="Length of each time-series sample.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.05,
        help="Observation noise standard deviation for the synthetic data.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20000,
        help="Number of synthetic samples to generate for training.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Logging interval in training steps.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=500,
        help="Evaluation interval in training steps.",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=8,
        help="Number of samples to use during evaluation and sampling.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=0.0,
        help="Classifier-free guidance scale during sampling.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient norm clipping value.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda, cuda:0, etc.).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
