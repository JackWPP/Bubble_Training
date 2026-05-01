"""Custom shape-preserving blocks for Bubble-YOLO11s.

The modules are intentionally independent of Ultralytics' internal Conv
wrappers so they remain stable across minor Ultralytics releases.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def choose_heads(channels: int, preferred: int | None = None) -> int:
    """Return a valid attention head count for a channel dimension."""
    if preferred and preferred > 0 and channels % preferred == 0:
        return preferred
    for heads in (8, 4, 2, 1):
        if channels % heads == 0:
            return heads
    return 1


class ConvBNAct(nn.Module):
    """Small Conv-BN-SiLU helper used by SSBRefine."""

    def __init__(
        self,
        c1: int,
        c2: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(c1, c2, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SSBRefine(nn.Module):
    """Shape-preserving Scale Switch inspired refinement block."""

    def __init__(
        self,
        c1: int,
        expansion: float = 1.0,
        shortcut: bool = True,
        gamma_init: float = 0.1,
    ) -> None:
        super().__init__()
        hidden = max(1, int(c1 * expansion))
        self.c1 = int(c1)
        self.proj_in = ConvBNAct(self.c1, hidden, kernel_size=3)
        self.dw = ConvBNAct(hidden, hidden, kernel_size=3, groups=hidden)
        self.proj_out = nn.Sequential(
            nn.Conv2d(hidden, self.c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.c1),
        )
        self.shortcut = shortcut
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.c1:
            raise RuntimeError(f"SSBRefine expected {self.c1} channels, got {x.shape[1]}")
        y = self.proj_out(self.dw(self.proj_in(x)))
        return x + self.gamma * y if self.shortcut else y


class LayerNorm2d(nn.Module):
    """LayerNorm over channel dimension for BCHW tensors."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True, unbiased=False)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class MDTA(nn.Module):
    """Multi-DConv Head Transposed Self-Attention."""

    def __init__(self, dim: int, num_heads: int | None = None, bias: bool = False) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_heads = choose_heads(self.dim, num_heads)
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))
        self.qkv = nn.Conv2d(self.dim, self.dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            self.dim * 3,
            self.dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if c != self.dim:
            raise RuntimeError(f"MDTA expected {self.dim} channels, got {c}")
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        head_dim = c // self.num_heads
        q = q.reshape(b, self.num_heads, head_dim, h * w)
        k = k.reshape(b, self.num_heads, head_dim, h * w)
        v = v.reshape(b, self.num_heads, head_dim, h * w)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.reshape(b, c, h, w)
        return self.project_out(out)


class GDFN(nn.Module):
    """Gated depthwise-conv feed-forward network."""

    def __init__(self, dim: int, expansion_factor: float = 2.0, bias: bool = False) -> None:
        super().__init__()
        hidden = max(1, int(dim * expansion_factor))
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden * 2,
            hidden * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dwconv(self.project_in(x))
        x1, x2 = x.chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class GLRB(nn.Module):
    """Global-local refinement block using MDTA and GDFN."""

    def __init__(
        self,
        c1: int,
        num_heads: int | None = None,
        expansion_factor: float = 2.0,
        bias: bool = False,
        gamma_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.c1 = int(c1)
        heads = choose_heads(self.c1, num_heads)
        self.norm1 = LayerNorm2d(self.c1)
        self.attn = MDTA(self.c1, num_heads=heads, bias=bias)
        self.norm2 = LayerNorm2d(self.c1)
        self.ffn = GDFN(self.c1, expansion_factor=expansion_factor, bias=bias)
        self.gamma1 = nn.Parameter(torch.tensor(float(gamma_init)))
        self.gamma2 = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.c1:
            raise RuntimeError(f"GLRB expected {self.c1} channels, got {x.shape[1]}")
        x = x + self.gamma1 * self.attn(self.norm1(x))
        x = x + self.gamma2 * self.ffn(self.norm2(x))
        return x
