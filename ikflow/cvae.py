# CVAE model for inverse kinematics
import math
from typing import Tuple
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from ikflow.config import DEVICE

from .my_utils import JointLimiter, mlp


@dataclass
class CVAEConfig:
    q_dim: int
    x_dim: int = 7
    z_dim: int = 32
    hidden: int = 512
    enc_layers: int = 3
    dec_layers: int = 3
    recon_loss: str = "l2"  # or "l1"
    beta_kl: float = 1.0
    use_joint_limits: bool = True


class CVAE(nn.Module):
    def __init__(self, cfg: CVAEConfig, joint_limiter: Optional[JointLimiter] = None):
        super().__init__()
        self.cfg = cfg
        self.joint_limiter = joint_limiter if cfg.use_joint_limits else None

        self.encoder = mlp(cfg.q_dim + cfg.x_dim, 2*cfg.z_dim, cfg.hidden, cfg.enc_layers)
        self.decoder = mlp(cfg.z_dim + cfg.x_dim, cfg.q_dim, cfg.hidden, cfg.dec_layers)

    def encode(self, q: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([q, x], dim=-1))
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar.clamp(min=-20, max=20)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out = self.decoder(torch.cat([z, x], dim=-1))
        if self.joint_limiter is not None:
            out = self.joint_limiter(out)
        return out

    def forward(self, q: torch.Tensor, x: torch.Tensor):
        mu, logvar = self.encode(q, x)
        z = self.reparameterize(mu, logvar)
        q_hat = self.decode(z, x)
        return q_hat, mu, logvar

    def loss(self, q: torch.Tensor, x: torch.Tensor):
        q_hat, mu, logvar = self.forward(q, x)
        if self.cfg.recon_loss == "l2":
            recon = F.mse_loss(q_hat, q)
        else:
            recon = F.l1_loss(q_hat, q)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        return recon + self.cfg.beta_kl * kl, {"recon": recon.detach(), "kl": kl.detach()}

    @torch.no_grad()
    def sample(self, x: torch.Tensor, n: Optional[int] = None) -> torch.Tensor:
        B = x.size(0)
        z = torch.randn(B, self.cfg.z_dim, device=x.device)
        return self.decode(z, x)

def cvae_model(cfg, joint_limits):
    joint_limits = torch.tensor(joint_limits, dtype=torch.float32).to(DEVICE)
    joint_limiter = JointLimiter(
        q_min=joint_limits[:, 0],
        q_max=joint_limits[:, 1]
    )
    joint_limiter.to(DEVICE)

    model = CVAE(cfg, joint_limiter)
    model.to(DEVICE)

    return model
