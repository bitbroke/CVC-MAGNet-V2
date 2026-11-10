"""Mamba S6 2D Context Module and Matched Conventional Convolution Context Module.

Designed for CVC-Mamba-Hybrid-01:
- 128 channels at stride-32 bottleneck (processes 24x24 = 576 tokens for 768px input).
- Four-way directional raster scan: row forward, row reverse, col forward, col reverse.
- Restores each ordering to (H, W), averages, and applies residual projection.
- Zero-initialized residual output projection:
    nn.init.zeros_(out_proj.weight)
    nn.init.zeros_(out_proj.bias)
  Guaranteeing that at update 0, Arm B and Arm C produce EXACT IDENTICAL predictions
  to the frozen S4 reference (Arm A).
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveScan1D(nn.Module):
    """Pure PyTorch implementation of Mamba S6 Selective State Space Scan."""

    def __init__(self, d_model: int = 128, d_state: int = 16, dt_rank: int = 8):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank

        # Project input x to dt, B, C
        self.x_proj = nn.Linear(d_model, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)

        # Log of continuous transition parameter A (initialized to HIPPO-like structure)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # Direct feedthrough D
        self.D = nn.Parameter(torch.ones(d_model))

        # Initialize dt_proj
        dt_init_std = dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        # Initialize dt bias to ~0.001 to 0.1
        dt = torch.exp(
            torch.rand(d_model) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        Args:
            u: [B, L, D] sequence of feature vectors.
        Returns:
            y: [B, L, D] sequence of output vectors.
        """
        B, L, D = u.shape

        # Compute input-dependent projections
        x_proj = self.x_proj(u)  # [B, L, dt_rank + 2*d_state]
        dt, B_t, C_t = torch.split(
            x_proj, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        dt = F.softplus(self.dt_proj(dt))  # [B, L, D]
        A = -torch.exp(self.A_log.float())  # [D, N]

        # Discretization
        # dt: [B, L, D] -> [B, L, D, 1]
        # A: [D, N] -> [1, 1, D, N]
        dA = torch.exp(dt.unsqueeze(-1) * A.view(1, 1, D, self.d_state))  # [B, L, D, N]
        # B_t: [B, L, N] -> [B, L, 1, N]
        dB = dt.unsqueeze(-1) * B_t.unsqueeze(2)  # [B, L, D, N]

        # Scan loop in float32 for numerical stability
        h = torch.zeros(B, D, self.d_state, device=u.device, dtype=torch.float32)
        ys = []
        u_f = u.float()

        for t in range(L):
            dA_t = dA[:, t]  # [B, D, N]
            dB_t = dB[:, t]  # [B, D, N]
            u_t = u_f[:, t].unsqueeze(-1)  # [B, D, 1]

            h = dA_t * h + dB_t * u_t  # [B, D, N]
            # y_t = C_t * h + D * u_t
            C_step = C_t[:, t].unsqueeze(1)  # [B, 1, N]
            y_t = torch.sum(h * C_step, dim=-1)  # [B, D]
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # [B, L, D]
        y = y + u_f * self.D.unsqueeze(0).unsqueeze(0)
        return y.to(u.dtype)


class FourWayMamba2DBlock(nn.Module):
    """2D Context Module using 4-directional Mamba S6 Scan.
    
    Directions:
    1. Row-major forward (H, W)
    2. Row-major reverse
    3. Column-major forward (W, H)
    4. Column-major reverse
    """

    def __init__(self, dim: int = 128, d_state: int = 16, dt_rank: int = 8):
        super().__init__()
        self.dim = dim
        self.norm = nn.GroupNorm(8, dim)
        self.in_proj = nn.Linear(dim, dim, bias=False)
        self.conv1d = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.act = nn.SiLU()

        # Shared selective scan across directions to conserve parameters
        self.scan = SelectiveScan1D(d_model=dim, d_state=d_state, dt_rank=dt_rank)

        # Output projection ZERO-INITIALIZED so initial residual is 0
        self.out_proj = nn.Linear(dim, dim, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: [B, C, H, W] feature map at stride 32.
        Returns:
            out: [B, C, H, W] with residual addition.
        """
        B, C, H, W = x.shape
        residual = x

        # Norm and project
        feat = self.norm(x)
        feat = feat.flatten(2).transpose(1, 2)  # [B, L=HW, C]
        feat = self.in_proj(feat)  # [B, L, C]

        # 1D Conv across sequence
        feat_conv = self.act(self.conv1d(feat.transpose(1, 2)).transpose(1, 2))  # [B, L, C]

        # 1. Row forward: [B, H*W, C]
        y_rf = self.scan(feat_conv)  # [B, L, C]

        # 2. Row reverse:
        feat_rr = torch.flip(feat_conv, dims=[1])
        y_rr = torch.flip(self.scan(feat_rr), dims=[1])

        # Reshape to (B, H, W, C) for column scans
        feat_2d = feat_conv.view(B, H, W, C)

        # 3. Col forward: transpose H and W -> [B, W, H, C] -> [B, W*H, C]
        feat_col = feat_2d.permute(0, 2, 1, 3).contiguous().view(B, W * H, C)
        y_cf = self.scan(feat_col).view(B, W, H, C).permute(0, 2, 1, 3).contiguous().view(B, H * W, C)

        # 4. Col reverse:
        feat_cr = torch.flip(feat_col, dims=[1])
        y_cr = torch.flip(self.scan(feat_cr), dims=[1]).view(B, W, H, C).permute(0, 2, 1, 3).contiguous().view(B, H * W, C)

        # Average four directions
        y_avg = (y_rf + y_rr + y_cf + y_cr) * 0.25  # [B, L, C]

        # Zero-initialized output projection
        out = self.out_proj(y_avg)  # [B, L, C]
        out_2d = out.transpose(1, 2).view(B, C, H, W)

        # Residual connection (zero-initialized => out_2d is initially 0 => output is residual)
        return residual + out_2d


class ConvContextBlock(nn.Module):
    """Conventional Convolution Context Module (Parameter-matched control for Arm C).
    
    Matches the parameter count and location of the Mamba block at the stride-32 bottleneck.
    Zero-initialized output projection ensures initial output matches Arm A exactly.
    """

    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim
        self.norm = nn.GroupNorm(8, dim)

        # Depthwise-separable residual Conv Block:
        # Parameter count: ~43,808 parameters (matched within 4.0% of Mamba's 42,112 parameters)
        self.dw_conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.pw_conv1 = nn.Conv2d(dim, 96, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(8, 96)
        self.relu1 = nn.ReLU(inplace=True)

        self.dw_conv2 = nn.Conv2d(96, 96, kernel_size=3, padding=1, groups=96, bias=False)
        self.pw_conv2 = nn.Conv2d(96, dim, kernel_size=1, bias=False)
        self.gn2 = nn.GroupNorm(8, dim)
        self.relu2 = nn.ReLU(inplace=True)

        # Zero-initialized projection
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        feat = self.norm(x)
        feat = self.dw_conv1(feat)
        feat = self.pw_conv1(feat)
        feat = self.gn1(feat)
        feat = self.relu1(feat)

        feat = self.dw_conv2(feat)
        feat = self.pw_conv2(feat)
        feat = self.gn2(feat)
        feat = self.relu2(feat)

        out = self.out_conv(feat)
        return residual + out
