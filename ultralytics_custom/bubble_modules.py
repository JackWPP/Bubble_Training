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


class MSLRefine(nn.Module):
    """Shape-preserving multi-scale local refinement block."""

    def __init__(
        self,
        c1: int,
        expansion: float = 0.5,
        shortcut: bool = True,
        gamma_init: float = 0.01,
    ) -> None:
        super().__init__()
        hidden = max(8, int(c1 * expansion))
        self.c1 = int(c1)
        self.reduce = ConvBNAct(self.c1, hidden, kernel_size=1)
        self.dw3 = ConvBNAct(hidden, hidden, kernel_size=3, groups=hidden)
        self.dw5 = ConvBNAct(hidden, hidden, kernel_size=5, groups=hidden)
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden * 2, self.c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.c1),
        )
        self.shortcut = shortcut
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.c1:
            raise RuntimeError(f"MSLRefine expected {self.c1} channels, got {x.shape[1]}")
        y = self.reduce(x)
        y = self.fuse(torch.cat((self.dw3(y), self.dw5(y)), dim=1))
        return x + self.gamma * y if self.shortcut else y


class P3CAGate(nn.Module):
    """Shape-preserving channel attention gate for the P3 detection feature."""

    def __init__(
        self,
        c1: int,
        reduction: int = 16,
        shortcut: bool = True,
        gamma_init: float = 0.01,
    ) -> None:
        super().__init__()
        self.c1 = int(c1)
        hidden = max(8, self.c1 // max(1, int(reduction)))
        self.fc1 = nn.Conv2d(self.c1, hidden, kernel_size=1, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, self.c1, kernel_size=1, bias=True)
        self.shortcut = shortcut
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.c1:
            raise RuntimeError(f"P3CAGate expected {self.c1} channels, got {x.shape[1]}")
        pooled = F.adaptive_avg_pool2d(x, 1) + F.adaptive_max_pool2d(x, 1)
        gate = torch.sigmoid(self.fc2(self.act(self.fc1(pooled))))
        scale = 1.0 + self.gamma * (gate - 0.5) * 2.0
        y = x * scale
        return y if self.shortcut else y - x


class P3SAGate(nn.Module):
    """Shape-preserving spatial attention gate for the P3 detection feature."""

    def __init__(
        self,
        c1: int,
        kernel_size: int = 7,
        shortcut: bool = True,
        gamma_init: float = 0.01,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("P3SAGate kernel_size must be odd")
        self.c1 = int(c1)
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)
        self.shortcut = shortcut
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.c1:
            raise RuntimeError(f"P3SAGate expected {self.c1} channels, got {x.shape[1]}")
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.amax(dim=1, keepdim=True)
        gate = torch.sigmoid(self.conv(torch.cat((avg_map, max_map), dim=1)))
        scale = 1.0 + self.gamma * (gate - 0.5) * 2.0
        y = x * scale
        return y if self.shortcut else y - x


class P3LCRefine(nn.Module):
    """Shape-preserving local-contrast residual refine block for the P3 feature."""

    def __init__(
        self,
        c1: int,
        kernel_size: int = 3,
        shortcut: bool = True,
        gamma_init: float = 1.0,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("P3LCRefine kernel_size must be odd")
        self.c1 = int(c1)
        self.pool = nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            count_include_pad=False,
        )
        self.dw = nn.Conv2d(
            self.c1,
            self.c1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=self.c1,
            bias=False,
        )
        self.shortcut = shortcut
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        nn.init.zeros_(self.dw.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.c1:
            raise RuntimeError(f"P3LCRefine expected {self.c1} channels, got {x.shape[1]}")
        contrast = x - self.pool(x)
        y = self.gamma * self.dw(contrast)
        return x + y if self.shortcut else y


class P3MLCRefine(nn.Module):
    """Multi-scale local-contrast residual refine block for the P3 feature."""

    def __init__(
        self,
        c1: int,
        shortcut: bool = True,
        gamma_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.c1 = int(c1)
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False)
        self.pool5 = nn.AvgPool2d(kernel_size=5, stride=1, padding=2, count_include_pad=False)
        self.dw3 = nn.Conv2d(self.c1, self.c1, kernel_size=3, padding=1, groups=self.c1, bias=False)
        self.dw5 = nn.Conv2d(self.c1, self.c1, kernel_size=5, padding=2, groups=self.c1, bias=False)
        self.fuse = nn.Conv2d(self.c1 * 2, self.c1, kernel_size=1, bias=False)
        self.shortcut = shortcut
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        nn.init.zeros_(self.dw3.weight)
        nn.init.zeros_(self.dw5.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.c1:
            raise RuntimeError(f"P3MLCRefine expected {self.c1} channels, got {x.shape[1]}")
        contrast3 = x - self.pool3(x)
        contrast5 = x - self.pool5(x)
        y = self.fuse(torch.cat((self.dw3(contrast3), self.dw5(contrast5)), dim=1))
        y = self.gamma * y
        return x + y if self.shortcut else y


class ECAGate(nn.Module):
    """Shape-preserving efficient channel attention gate."""

    def __init__(
        self,
        c1: int,
        kernel_size: int = 5,
        shortcut: bool = True,
        gamma_init: float = 0.01,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("ECAGate kernel_size must be odd")
        self.c1 = int(c1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=int(kernel_size),
            padding=int(kernel_size) // 2,
            bias=True,
        )
        self.shortcut = shortcut
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.c1:
            raise RuntimeError(f"ECAGate expected {self.c1} channels, got {x.shape[1]}")
        pooled = F.adaptive_avg_pool2d(x, 1).flatten(2).transpose(1, 2)
        gate = torch.sigmoid(self.conv(pooled)).transpose(1, 2).unsqueeze(-1)
        y = x * (1.0 + self.gamma * (gate - 0.5) * 2.0)
        return y if self.shortcut else y - x


class CoordGate(nn.Module):
    """Shape-preserving coordinate attention gate with near-identity init."""

    def __init__(
        self,
        c1: int,
        reduction: int = 32,
        shortcut: bool = True,
        gamma_init: float = 0.01,
    ) -> None:
        super().__init__()
        self.c1 = int(c1)
        hidden = max(8, self.c1 // max(1, int(reduction)))
        self.conv1 = nn.Conv2d(self.c1, hidden, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = nn.SiLU(inplace=True)
        self.conv_h = nn.Conv2d(hidden, self.c1, kernel_size=1, bias=True)
        self.conv_w = nn.Conv2d(hidden, self.c1, kernel_size=1, bias=True)
        self.shortcut = shortcut
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        nn.init.zeros_(self.conv_h.weight)
        nn.init.zeros_(self.conv_h.bias)
        nn.init.zeros_(self.conv_w.weight)
        nn.init.zeros_(self.conv_w.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.c1:
            raise RuntimeError(f"CoordGate expected {self.c1} channels, got {x.shape[1]}")
        h = x.shape[2]
        pool_h = x.mean(dim=3, keepdim=True)
        pool_w = x.mean(dim=2, keepdim=True).transpose(2, 3)
        y = self.act(self.bn1(self.conv1(torch.cat((pool_h, pool_w), dim=2))))
        y_h, y_w = torch.split(y, [h, y.shape[2] - h], dim=2)
        gate_h = torch.sigmoid(self.conv_h(y_h))
        gate_w = torch.sigmoid(self.conv_w(y_w.transpose(2, 3)))
        gate = gate_h * gate_w
        y = x * (1.0 + self.gamma * (gate - 0.25) * 4.0)
        return y if self.shortcut else y - x


class SimAMGate(nn.Module):
    """Shape-preserving SimAM-style parameter-light 3D attention gate."""

    def __init__(
        self,
        c1: int,
        lambda_e: float = 1e-4,
        shortcut: bool = True,
        gamma_init: float = 0.005,
    ) -> None:
        super().__init__()
        self.c1 = int(c1)
        self.lambda_e = float(lambda_e)
        self.shortcut = shortcut
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.c1:
            raise RuntimeError(f"SimAMGate expected {self.c1} channels, got {x.shape[1]}")
        mean = x.mean(dim=(2, 3), keepdim=True)
        diff = (x - mean).pow(2)
        denom = 4.0 * (diff.mean(dim=(2, 3), keepdim=True) + self.lambda_e)
        attention = torch.sigmoid(diff / denom + 0.5)
        y = x * attention
        return x + self.gamma * (y - x) if self.shortcut else y


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
