import math
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# (optional) joint-limit squashing to keep outputs feasible
class JointLimiter(nn.Module):
    """Squash unconstrained outputs to joint limits via tanh mapping."""
    def __init__(self, q_min: torch.Tensor, q_max: torch.Tensor):
        super().__init__()
        assert q_min.shape == q_max.shape
        self.register_buffer("q_min", q_min)
        self.register_buffer("q_max", q_max)
        self.register_buffer("q_mid", 0.5 * (q_min + q_max))
        self.register_buffer("half_span", 0.5 * (q_max - q_min))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y is unconstrained; map with tanh to [q_min, q_max]
        return self.q_mid + self.half_span * torch.tanh(y)
    
def mlp(in_dim: int, out_dim: int, hidden: int, layers: int) -> nn.Sequential:
    mods = [nn.Linear(in_dim, hidden), nn.SiLU()]
    for _ in range(layers-1):
        mods += [nn.Linear(hidden, hidden), nn.SiLU()]
    mods += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*mods)
    