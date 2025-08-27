# Diffusion model for inverse kinematics
import math
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from ikflow.config import DEVICE


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


# ---------- Residual Block (time-conditioning) ----------
class ResidualBlock(nn.Module):
    """
    Residual block that integrates timestep embedding with q^t embedding.
    Follows RMSNorm + MLP + skip-connection style.
    """
    def __init__(self, d_model=512):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.mlp_q = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.norm2 = nn.RMSNorm(d_model)    # TODO: apply norm to time embedding or not?
        self.mlp_t = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.out_norm = nn.RMSNorm(d_model)
        self.out_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, q_tok, t_emb):
        """
        q_tok: [B, d_model] (noised q embedding)
        t_emb: [B, d_model] (timestep embedding)
        """
        # process q
        q_res = self.mlp_q(self.norm1(q_tok))
        t_res = self.mlp_t(self.norm2(t_emb))
        out = q_res + t_res

        return q_tok + self.out_mlp(self.out_norm(out))


# ---------- Transformer Block (pre-norm cross-attention) ----------
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim_q: int, dim_kv: int, dim_out: int,
                 num_heads: int = 8, dropout: float = 0.0):
        """
        Multi-head cross-attention:
            Query: from target sequence (dim_q)
            Key/Value: from context sequence (dim_kv)
            Output: projected to dim_out
        Args:
            dim_q: query input dimension
            dim_kv: key/value input dimension
            dim_out: output dimension (usually = dim_q)
            num_heads: number of attention heads
            dropout: dropout prob for attention weights
        """
        super().__init__()
        assert dim_out % num_heads == 0, "dim_out must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads
        self.scale = self.head_dim ** -0.5

        # projections
        self.q_proj = nn.Linear(dim_q, dim_out)
        self.k_proj = nn.Linear(dim_kv, dim_out)
        self.v_proj = nn.Linear(dim_kv, dim_out)
        self.out_proj = nn.Linear(dim_out, dim_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            q: (B, Nq, dim_q) queries
            kv: (B, Nk, dim_kv) keys/values
            mask: (B, 1, Nk) or (B, Nq, Nk), optional
        Returns:
            out: (B, Nq, dim_out)
        """
        B, Nq, _ = q.size()
        Nk = kv.size(1)

        # linear projections
        q_proj = self.q_proj(q)      # (B, Nq, dim_out)
        k_proj = self.k_proj(kv)     # (B, Nk, dim_out)
        v_proj = self.v_proj(kv)     # (B, Nk, dim_out)

        # split heads
        q_proj = q_proj.view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Nq, Hd)
        k_proj = k_proj.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Nk, Hd)
        v_proj = v_proj.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Nk, Hd)

        # attention scores
        attn = torch.matmul(q_proj, k_proj.transpose(-2, -1)) * self.scale  # (B,H,Nq,Nk)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # weighted sum
        out = torch.matmul(attn, v_proj)  # (B,H,Nq,Hd)
        out = out.transpose(1, 2).contiguous().view(B, Nq, -1)  # (B, Nq, dim_out)

        return self.out_proj(out)
    

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, ff_hidden_dim=1024):
        super().__init__()
        self.norm_q = nn.RMSNorm(dim)     # TODO: LayerNorm or RMSNorm?
        self.norm_x = nn.RMSNorm(dim)     # TODO: Apply norm to x (key & value) or not?
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

        self.norm_out = nn.RMSNorm(dim)  # Output norm
        # (PointWise) Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_hidden_dim),
            nn.GELU(),
            nn.Linear(ff_hidden_dim, dim),
        )

    def forward(self, q, x):
        # Pre-norm
        q_norm = self.norm_q(q)
        x_norm = self.norm_x(x)

        # Cross-attention: q queries x
        attn_out, _ = self.attn(query=q_norm, key=x_norm, value=x_norm) # attn_out = x_norm if seq_len=1
        q = q + attn_out

        # Feed-forward with residual
        q = q + self.ff(self.norm_out(q))
        return q


class TransformerDiffusion(nn.Module):
    def __init__(self, q_dim, x_dim, dim=512, num_heads=8, num_layers=8, ff_hidden_dim=1024):
        super().__init__()
        self.q_embed = nn.Linear(q_dim, dim)
        self.t_embed = nn.Linear(1, dim)
        self.x_embed = nn.Linear(x_dim, dim)

        self.channels = q_dim

        # 8 stacked Residual + Transformer blocks
        self.residual_blocks = nn.ModuleList([ResidualBlock(dim) for _ in range(num_layers)])
        self.transformer_blocks = nn.ModuleList([TransformerBlock(dim, num_heads, ff_hidden_dim) for _ in range(num_layers)])

        self.out_norm = nn.RMSNorm(dim)
        self.out_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, q_dim)
        )

    def forward(self, q_t, t, x):
        """
        q_t: (B, q_dim)    - noised joint configuration
        t:   (B, 1)        - diffusion timestep
        x:   (B, x_dim)    - end-effector pose
        """
        # Embeddings
        q_tok = self.q_embed(q_t)
        t_emb = self.t_embed(t)
        q_tok = q_tok.unsqueeze(1)  # (B, 1, dim)
        t_emb = t_emb.unsqueeze(1)
        x_emb = self.x_embed(x)

        if len(x_emb.shape) == 2:
            x_emb = x_emb.unsqueeze(1)

        # Apply 8 Residual + Transformer blocks
        for res, trans in zip(self.residual_blocks, self.transformer_blocks):
            q_tok = res(q_tok, t_emb)
            q_tok = trans(q_tok, x_emb)

        q_tok = q_tok.squeeze(1)
        return self.out_mlp(self.out_norm(q_tok))
    

def transformer_diffusion_model(cfg):
    model = TransformerDiffusion(
        q_dim=cfg.q_dim,
        x_dim=cfg.x_dim,
        dim=cfg.hidden,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        ff_hidden_dim=cfg.ff_hidden_dim
    )
    model.to(DEVICE)

    return model


if __name__ == '__main__':
    batch_size = 100
    device = 'cuda:0'
    model = TransformerDiffusion(q_dim=6, x_dim=7, dim=512, num_heads=8, num_layers=8, ff_hidden_dim=1024)
    q_t = torch.randn(batch_size, 6).to(device)  # Noised joint config
    t = torch.randint(0, 100, (batch_size, 1)).float().to(device)  # Random timesteps
    x = torch.randn(batch_size, 7).to(device)  # End-effector poses (sequence length 10)

    model.to(device)
    eps = model(q_t, t, x)

    breakpoint()
