"""Pure PyTorch Mamba S6 2D Bottleneck Block for Windows/CUDA.

Implements the selective state-space model (S6) from Gu & Dao (2023)
tailored for 2D spatial feature map bottlenecks:
- Input: [B, C, H, W] feature map at Stride 32 (e.g. C=128, H=24, W=24 at 768px input)
- Bidirectional or 1D-flattened sequential spatial scan across L = H * W tokens
- Selective input-dependent discretization: Delta, B, C
- Discretized state updates with diagonal state matrix A
- Gated SiLU branch and linear projection
- Residual connection around the block
- 100% pure PyTorch: no custom CUDA compilation required, fully compatible with Windows.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SelectiveScan1D(nn.Module):
    """Discretized Selective State Space (S6) core recurrence in pure PyTorch."""

    def __init__(self, d_inner: int, d_state: int = 16):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state

        # Log of diagonal state transition A (initialized to HIPPO-like decay)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))  # [d_inner, d_state]

        # Parameter D (skip connection)
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of S6 recurrence.

        Args:
            u: [B, L, d_inner] input sequence
            delta: [B, L, d_inner] input-dependent time step
            B: [B, L, d_state] input-dependent B matrix
            C: [B, L, d_state] input-dependent C matrix

        Returns:
            y: [B, L, d_inner] output sequence
        """
        b, l, d = u.shape
        d_state = self.d_state

        # Continuous to discrete discretization
        # A_bar = exp(delta * (-exp(A_log)))
        A = -torch.exp(self.A_log.float())  # [d, d_state]
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # [B, L, d, d_state]
        deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)  # [B, L, d, d_state]

        # Sequential scan over sequence length L (L = 576 at stride 32 is fast in PyTorch)
        # Using loop with torch.jit or simple unrolled accumulation
        # Memory-efficient recurrence
        h = torch.zeros(b, d, d_state, device=u.device, dtype=u.dtype)
        ys = []

        for t in range(l):
            h = deltaA[:, t] * h + deltaB_u[:, t]  # [B, d, d_state]
            # y_t = (h * C_t).sum(-1)
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # [B, d]
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # [B, L, d]
        y = y + u * self.D.view(1, 1, -1)
        return y


class Mamba2DBottleneckBlock(nn.Module):
    """Residual 2D Mamba block for feature map bottlenecks."""

    def __init__(
        self,
        dim: int = 128,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        dt_rank: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(expand * dim)
        self.dt_rank = dt_rank or max(dim // 16, 16)

        # Norm before Mamba
        self.norm = nn.GroupNorm(num_groups=8, num_channels=dim)

        # In-projection (projects to u and z branches)
        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=False)

        # 1D Depthwise Conv across spatial sequence
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            padding=(d_conv - 1) // 2,
            groups=self.d_inner,
        )

        # Projections for selective parameters
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # S6 scan core
        self.ssm = SelectiveScan1D(d_inner=self.d_inner, d_state=self.d_state)

        # Out-projection back to dim
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, C, H, W] spatial feature map

        Returns:
            out: [B, C, H, W] refined feature map with residual connection
        """
        residual = x
        b, c, h, w = x.shape
        norm_x = self.norm(x)

        # Flatten spatial dimensions to sequence: [B, L, C] where L = H * W
        seq_x = rearrange(norm_x, "b c h w -> b (h w) c")

        # Project to (u, z)
        u_z = self.in_proj(seq_x)  # [B, L, 2 * d_inner]
        u, z = u_z.chunk(2, dim=-1)  # each [B, L, d_inner]

        # 1D Conv across sequence: [B, d_inner, L]
        u_conv = rearrange(u, "b l d -> b d l")
        u_conv = F.silu(self.conv1d(u_conv))
        u_conv = rearrange(u_conv, "b d l -> b l d")

        # Selective parameters
        x_dbl = self.x_proj(u_conv)  # [B, L, dt_rank + 2*d_state]
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))  # [B, L, d_inner]

        # SSM forward
        y = self.ssm(u_conv, delta, B, C)  # [B, L, d_inner]

        # Gating with z
        y = y * F.silu(z)

        # Out projection
        out_seq = self.out_proj(y)  # [B, L, dim]

        # Reshape back to spatial feature map: [B, dim, H, W]
        out_map = rearrange(out_seq, "b (h w) c -> b c h w", h=h, w=w)

        return residual + out_map
