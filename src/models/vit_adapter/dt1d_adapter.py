"""Final DT1D-Adapter for spatial feature maps and ViT patch tokens.

The public implementation follows the final paper architecture only.  It uses
one 17-tap depthwise convolution per enabled spatial axis.  Each channel group
shares a symmetric base kernel supported at 0, ±1, ±2, ±4, and ±8, with a
restricted zero-mean channel-dependent correction.  Height/width kernels are
jointly projected in L1 and injected through a small learned residual gate.

There are no legacy shifted-kernel branches, dilation routers, sample-adaptive
routers, or default pointwise mixers in this module.
"""
from __future__ import annotations

import math
from math import gcd
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_DETAIL_COMPONENTS = {"both", "offset4", "offset8", "none"}
_DETAIL_BASES = {"orth", "raw"}
_GATE_MODES = {"learned", "fixed"}
_VALID_OFFSETS = (1, 2, 4, 8)


def _parse_offsets(value: str | Sequence[int] | None) -> Tuple[int, ...]:
    if value is None:
        return _VALID_OFFSETS
    if isinstance(value, str):
        raw = [int(v) for v in value.replace(";", ",").replace(" ", ",").split(",") if v.strip()]
    else:
        raw = [int(v) for v in value]
    offsets = tuple(v for v in _VALID_OFFSETS if v in set(raw))
    if not offsets:
        raise ValueError(f"active_offsets must contain at least one of {_VALID_OFFSETS}")
    return offsets


class DT1DAdapter(nn.Module):
    """Final spatial DT1D adapter operating on BCHW tensors."""

    method_name = "DT1D-Adapter"

    def __init__(
        self,
        C: int,
        *,
        axis: str = "hw",
        group_size: int = 16,
        residual_scale: float = 1.0,
        gate_init: float = 0.01,
        gate_mode: str = "learned",
        padding_mode: str = "replicate",
        contrast_split: int = 8,
        detail_basis: str = "orth",
        detail_components: str = "offset4",
        active_offsets: str | Sequence[int] | None = None,
        use_pointwise: bool = False,
        pointwise_ratio: int = 32,
        pointwise_groups: int = 4,
        use_bn: bool = False,
        cache_kernel: bool = False,
        project_l1: bool = True,
    ) -> None:
        super().__init__()
        if C <= 0:
            raise ValueError(f"C must be positive, got {C}")
        if axis not in {"h", "w", "hw"}:
            raise ValueError(f"axis must be h, w, or hw, got {axis!r}")
        if padding_mode not in {"reflect", "replicate", "zeros", "constant"}:
            raise ValueError(f"Unsupported padding_mode={padding_mode!r}")
        detail_basis = str(detail_basis).lower()
        detail_components = str(detail_components).lower()
        gate_mode = str(gate_mode).lower()
        if detail_basis not in _DETAIL_BASES:
            raise ValueError(f"detail_basis must be one of {_DETAIL_BASES}")
        if detail_components not in _DETAIL_COMPONENTS:
            raise ValueError(f"detail_components must be one of {_DETAIL_COMPONENTS}")
        if gate_mode not in _GATE_MODES:
            raise ValueError(f"gate_mode must be one of {_GATE_MODES}")

        self.C = int(C)
        self.axis = axis
        self.axis_names = tuple(a for a in ("h", "w") if a in axis)
        self.num_axes = len(self.axis_names)
        self.group_size = max(1, int(group_size))
        self.num_groups = math.ceil(self.C / self.group_size)
        self.residual_scale = float(residual_scale)
        self.padding_mode = "constant" if padding_mode == "zeros" else padding_mode
        self.contrast_split = max(1, int(contrast_split))
        self.detail_basis = detail_basis
        self.detail_components = detail_components
        self.active_offsets = _parse_offsets(active_offsets)
        self.base_offsets = (0,) + self.active_offsets
        self.gate_mode = gate_mode
        self.cache_kernel = bool(cache_kernel)
        self.project_l1 = bool(project_l1)
        self.use_pointwise = bool(use_pointwise)

        self.base_coefficients = nn.Parameter(
            torch.zeros(self.num_axes, self.num_groups, len(self.base_offsets))
        )
        with torch.no_grad():
            self.base_coefficients[..., 0].fill_(1.0 / max(1, self.num_axes))

        active_detail = self._active_component_indices(detail_components)
        self.register_buffer("detail_component_indices", torch.tensor(active_detail, dtype=torch.long), persistent=True)
        self.detail_coefficients = nn.Parameter(
            torch.zeros(self.num_axes, self.num_groups, len(active_detail))
        )

        contrast, valid = self._make_channel_contrast()
        self.register_buffer("channel_contrast", contrast, persistent=True)
        self.register_buffer("valid_contrast_group", valid, persistent=True)
        self.register_buffer("detail_basis_functions", self._detail_basis_functions(detail_basis), persistent=True)

        if gate_mode == "learned":
            self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        else:
            self.register_buffer("gate", torch.tensor(float(gate_init)), persistent=True)

        if self.use_pointwise:
            hidden = max(1, self.C // max(1, int(pointwise_ratio)))
            groups = max(1, min(int(pointwise_groups), self.C, hidden))
            groups = gcd(groups, self.C)
            groups = gcd(groups, hidden) or 1
            self.pointwise = nn.Sequential(
                nn.Conv2d(self.C, hidden, 1, groups=groups, bias=False),
                nn.BatchNorm2d(hidden) if use_bn else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, self.C, 1, groups=groups, bias=False),
                nn.BatchNorm2d(self.C) if use_bn else nn.Identity(),
            )
        else:
            self.pointwise = nn.Identity()

        self.register_buffer("_cached_kernels", torch.empty(0), persistent=False)
        self.is_dt1d_adapter = True

    @staticmethod
    def _active_component_indices(detail_components: str) -> Tuple[int, ...]:
        if detail_components == "both": return (0, 1)
        if detail_components == "offset4": return (0,)
        if detail_components == "offset8": return (1,)
        return tuple()

    def _make_channel_contrast(self) -> Tuple[torch.Tensor, torch.Tensor]:
        contrast = torch.zeros(self.C, dtype=torch.float32)
        valid = torch.zeros(self.num_groups, dtype=torch.float32)
        start = 0
        for group in range(self.num_groups):
            n = min(self.group_size, self.C - start)
            n1 = min(self.contrast_split, max(1, n // 2))
            n2 = n - n1
            if n1 > 0 and n2 > 0:
                pos = math.sqrt(n2 / (n1 * (n1 + n2)))
                neg = -math.sqrt(n1 / (n2 * (n1 + n2)))
                contrast[start:start+n1] = pos
                contrast[start+n1:start+n] = neg
                valid[group] = 1.0
            start += n
        return contrast, valid

    @staticmethod
    def _second_difference(radius: int) -> torch.Tensor:
        atom = torch.zeros(17, dtype=torch.float64)
        atom[8-radius] = 1.0
        atom[8+radius] = 1.0
        atom[8] = -2.0
        return atom

    @classmethod
    def _detail_basis_functions(cls, basis: str) -> torch.Tensor:
        d4 = cls._second_difference(4)
        d8 = cls._second_difference(8)
        if basis == "raw":
            return torch.stack((d4, d8)).float()
        b4 = d4 / torch.linalg.vector_norm(d4)
        d8_perp = d8 - torch.dot(d8, b4) * b4
        b8 = d8_perp / torch.linalg.vector_norm(d8_perp)
        return torch.stack((b4, b8)).float()

    def _group_index(self, device: torch.device) -> torch.Tensor:
        return (torch.arange(self.C, device=device) // self.group_size).clamp_max(self.num_groups - 1)

    def build_kernels(self, device: torch.device, dtype: torch.dtype, *, project: bool | None = None) -> torch.Tensor:
        """Return kernels with shape (num_axes, C, 1, 17)."""
        if project is None:
            project = self.project_l1
        group_idx = self._group_index(device)
        beta = self.base_coefficients.to(device=device, dtype=dtype)[:, group_idx, :]
        kernel = torch.zeros(self.num_axes, self.C, 17, device=device, dtype=dtype)
        kernel[..., 8] = beta[..., 0]
        for coefficient, offset in enumerate(self.active_offsets, start=1):
            kernel[..., 8-offset] = beta[..., coefficient]
            kernel[..., 8+offset] = beta[..., coefficient]

        if self.detail_coefficients.shape[-1] > 0:
            valid = self.valid_contrast_group.to(device=device, dtype=dtype)
            eta = self.detail_coefficients.to(device=device, dtype=dtype)
            eta = eta * valid.view(1, self.num_groups, 1)
            eta_channel = eta[:, group_idx, :]
            contrast = self.channel_contrast.to(device=device, dtype=dtype)
            basis = self.detail_basis_functions.to(device=device, dtype=dtype)
            basis = basis[self.detail_component_indices.to(device=device)]
            kernel = kernel + torch.einsum("acr,rk->ack", eta_channel * contrast.view(1, self.C, 1), basis)

        if project:
            joint_l1 = kernel.abs().sum(dim=-1).sum(dim=0)
            scale = torch.maximum(joint_l1, torch.ones_like(joint_l1))
            kernel = kernel / scale.view(1, self.C, 1)
        return kernel.unsqueeze(2)

    def _pad(self, x: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
        if pad_h == 0 and pad_w == 0: return x
        if self.padding_mode == "constant":
            return F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="constant", value=0.0)
        mode = self.padding_mode
        if mode == "reflect":
            h, w = x.shape[-2:]
            if (pad_h >= h and pad_h > 0) or (pad_w >= w and pad_w > 0):
                mode = "replicate"
        return F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode=mode)

    def _conv_axis(self, x: torch.Tensor, axis_name: str, kernel: torch.Tensor) -> torch.Tensor:
        k = int(kernel.shape[-1]); radius = k // 2
        if axis_name == "h":
            x = self._pad(x, radius, 0)
            return F.conv2d(x, kernel.view(self.C, 1, k, 1), groups=self.C)
        if axis_name == "w":
            x = self._pad(x, 0, radius)
            return F.conv2d(x, kernel.view(self.C, 1, 1, k), groups=self.C)
        raise ValueError(f"Unknown axis {axis_name!r}")

    @torch.no_grad()
    def prepare_for_inference(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        ref = self.base_coefficients
        device = ref.device if device is None else device
        dtype = ref.dtype if dtype is None else dtype
        self._cached_kernels = self.build_kernels(device, dtype).detach()

    def clear_inference_cache(self) -> None:
        self._cached_kernels = torch.empty(0, device=self.base_coefficients.device)

    def train(self, mode: bool = True):
        if mode: self.clear_inference_cache()
        return super().train(mode)

    def _kernels_for(self, x: torch.Tensor) -> torch.Tensor:
        if self.cache_kernel and not self.training:
            if self._cached_kernels.numel() and self._cached_kernels.device == x.device and self._cached_kernels.dtype == x.dtype:
                return self._cached_kernels
            kernels = self.build_kernels(x.device, x.dtype)
            self._cached_kernels = kernels.detach()
            return kernels
        return self.build_kernels(x.device, x.dtype)

    def parameter_count_breakdown(self) -> Dict[str, int]:
        base = self.base_coefficients.numel()
        detail = self.detail_coefficients.numel()
        gate = self.gate.numel() if isinstance(self.gate, nn.Parameter) else 0
        pointwise = sum(p.numel() for p in self.pointwise.parameters())
        return {"shared_base": int(base), "channel_correction": int(detail), "learned_gate": int(gate), "pointwise": int(pointwise), "total": int(base+detail+gate+pointwise)}

    @property
    def convolution_calls_per_forward(self) -> int:
        return self.num_axes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"DT1DAdapter expects BCHW input, got {tuple(x.shape)}")
        if x.shape[1] != self.C:
            raise ValueError(f"Channel mismatch: adapter C={self.C}, input C={x.shape[1]}")
        kernels = self._kernels_for(x)
        response = torch.zeros_like(x)
        for axis_index, axis_name in enumerate(self.axis_names):
            response = response + self._conv_axis(x, axis_name, kernels[axis_index])
        response = self.pointwise(response)
        return x + self.residual_scale * self.gate.to(dtype=x.dtype, device=x.device) * response


class DT1DTokenAdapter(nn.Module):
    """Apply DT1D to ViT patch tokens while leaving prefix/class tokens unchanged."""

    def __init__(self, embed_dim: int, grid_size: Tuple[int, int], num_prefix_tokens: int = 1, **kwargs) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.grid_size = (int(grid_size[0]), int(grid_size[1]))
        self.num_prefix_tokens = int(num_prefix_tokens)
        if min(self.grid_size) <= 0:
            raise ValueError(f"Invalid grid_size={grid_size}")
        if self.num_prefix_tokens < 0:
            raise ValueError("num_prefix_tokens must be non-negative")
        self.spatial_adapter = DT1DAdapter(C=self.embed_dim, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"DT1DTokenAdapter expects BND tokens, got {tuple(x.shape)}")
        B, N, D = x.shape
        if D != self.embed_dim:
            raise ValueError(f"Embedding mismatch: got {D}, expected {self.embed_dim}")
        gh, gw = self.grid_size
        expected = self.num_prefix_tokens + gh * gw
        if N != expected:
            raise ValueError(f"N={N}, expected {expected} for grid {gh}x{gw} with {self.num_prefix_tokens} prefix token(s)")
        prefix = x[:, :self.num_prefix_tokens, :]
        patches = x[:, self.num_prefix_tokens:, :]
        fmap = patches.transpose(1, 2).reshape(B, D, gh, gw)
        fmap = self.spatial_adapter(fmap)
        patches = fmap.reshape(B, D, gh * gw).transpose(1, 2)
        return torch.cat((prefix, patches), dim=1) if self.num_prefix_tokens else patches

    def parameter_count_breakdown(self) -> Dict[str, int]:
        return self.spatial_adapter.parameter_count_breakdown()


__all__ = ["DT1DAdapter", "DT1DTokenAdapter"]
