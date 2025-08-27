# Diffusion model for inverse kinematics
import math
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .my_utils import JointLimiter, mlp

# ---------------------------
# Embeddings & Blocks
# ---------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: int = 10_000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.proj = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) integer or float timesteps
        returns: (B, dim)
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(0, half, device=t.device) / half
        )  # (half,)
        args = t[:, None] * freqs[None, :]  # (B, half)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, dim)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0,1))
        return self.proj(emb)


class FiLM(nn.Module):
    """Feature-wise linear modulation from a conditioning vector."""
    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim*2),
            nn.SiLU(),
            nn.Linear(hidden_dim*2, hidden_dim*2)
        )

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.net(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return h * (1 + gamma) + beta


class PreNormResidual(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fn(self.norm(x))


# ---------------------------
# Diffusion Model (DDPM-style)
# ---------------------------

@dataclass
class DiffusionConfig:
    q_dim: int
    x_dim: int = 7
    hidden: int = 512
    layers: int = 3
    cond_hidden: int = 512
    timesteps: int = 1000
    beta_schedule: str = "cosine"  # "linear" | "cosine"
    loss_type: str = "l2"          # "l2" or "l1"


def make_beta_schedule(T: int, kind: str = "cosine") -> torch.Tensor:
    if kind == "linear":
        beta = torch.linspace(1e-4, 0.02, T)
    elif kind == "cosine":
        # Nichol & Dhariwal cosine schedule in alphas
        s = 0.008
        steps = torch.arange(T+1, dtype=torch.float64)
        alphas_cumprod = torch.cos(((steps/T + s)/(1+s)) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        beta = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        beta = beta.clamp(1e-8, 0.999)
        beta = beta.float()
    else:
        raise ValueError("unknown schedule")
    return beta


class CondDenoiser(nn.Module):
    """
    Lightweight MLP denoiser with FiLM from (x, t).
    Predicts noise ε given (q_t, x, t).
    """
    def __init__(self, q_dim: int, x_dim: int, hidden: int, layers: int, cond_hidden: int):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(cond_hidden)
        self.cond_proj = mlp(x_dim + cond_hidden, cond_hidden, cond_hidden, 2)
        self.backbone = nn.Sequential(
            PreNormResidual(hidden, mlp(hidden, hidden, hidden, 1)),
            PreNormResidual(hidden, mlp(hidden, hidden, hidden, 1)),
            PreNormResidual(hidden, mlp(hidden, hidden, hidden, 1)),
        )
        self.in_proj  = nn.Linear(q_dim, hidden)
        self.film     = FiLM(cond_hidden, hidden)
        self.out_proj = nn.Linear(hidden, q_dim)
        self.mid = nn.Sequential(*[
            PreNormResidual(hidden, mlp(hidden, hidden, hidden, 1))
            for _ in range(max(0, layers-1))
        ])

    def forward(self, q_t: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (B, x_dim); t: (B,)
        temb = self.time_emb(t.float())         # (B, cond_hidden)
        c = self.cond_proj(torch.cat([x, temb], dim=-1))  # (B, cond_hidden)
        h = self.in_proj(q_t)
        h = self.film(h, c)
        h = self.backbone(h)
        h = self.mid(h)
        return self.out_proj(h)


class FiLMDiffusion(nn.Module):
    def __init__(self, cfg: DiffusionConfig, joint_limiter: Optional[JointLimiter] = None):
        super().__init__()
        self.cfg = cfg
        self.eps_model = CondDenoiser(cfg.q_dim, cfg.x_dim, cfg.hidden, cfg.layers, cfg.cond_hidden)
        beta = make_beta_schedule(cfg.timesteps, cfg.beta_schedule)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", 1.0 - beta)
        self.register_buffer("alpha_bar", torch.cumprod(1.0 - beta, dim=0))
        self.joint_limiter = joint_limiter  # used at sampling time

    def q_sample(self, q0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """
        q_t = sqrt(alpha_bar_t) * q0 + sqrt(1 - alpha_bar_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(q0)
        a_bar = self.alpha_bar[t].view(-1, 1)
        return a_bar.sqrt() * q0 + (1 - a_bar).sqrt() * noise, noise

    def p_losses(self, q0: torch.Tensor, x: torch.Tensor, t: torch.Tensor):
        q_t, noise = self.q_sample(q0, t)
        eps_pred = self.eps_model(q_t, x, t)
        if self.cfg.loss_type == "l2":
            loss = F.mse_loss(eps_pred, noise)
        else:
            loss = F.l1_loss(eps_pred, noise)
        return loss

    @torch.no_grad()
    def p_sample(self, q_t: torch.Tensor, x: torch.Tensor, t: int) -> torch.Tensor:
        """
        One reverse step q_{t-1} from q_t.
        """
        beta_t = self.beta[t]
        alpha_t = self.alpha[t]
        a_bar_t = self.alpha_bar[t]
        eps = self.eps_model(q_t, x, torch.full((q_t.shape[0],), t, device=q_t.device))
        mean = (1 / alpha_t.sqrt()) * (q_t - ((1 - alpha_t) / (1 - a_bar_t).sqrt()) * eps)
        if t > 0:
            z = torch.randn_like(q_t)
            return mean + beta_t.sqrt() * z
        else:
            return mean

    @torch.no_grad()
    def sample(self, x: torch.Tensor, n_steps: Optional[int] = None) -> torch.Tensor:
        """
        x: (B, x_dim) conditioning. Returns (B, q_dim)
        """
        T = n_steps or self.cfg.timesteps
        B = x.shape[0]
        q_t = torch.randn(B, self.cfg.q_dim, device=x.device)
        for t in reversed(range(T)):
            q_t = self.p_sample(q_t, x, t)
        if self.joint_limiter is not None:
            q_t = self.joint_limiter(q_t)
        return q_t
