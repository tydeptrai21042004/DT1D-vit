"""Final WHC-Compact-DT1D adapter for ViT patch-token grids.

The proposal keeps the compact symmetric DT1D support {1,2,4}, then fuses a
normalized weighted h-Hartley-cosine shift pair into the axial kernels:

    k_W = (1-lambda) k + lambda/2 (S_-p k + S_+p k),

with p=2 in the final proposal.  lambda is learned once per transformer block,
while the outer residual injection gate is fixed at gamma=0.01.  The final H/W
kernels are jointly projected in L1 and executed with one depthwise convolution
per enabled spatial axis.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dt1d_adapter import DT1DAdapter, _parse_offsets

_LAMBDA_MODES = {"learned", "fixed", "off"}
_LAMBDA_SCOPES = {"block", "axis"}
_SHIFT_NORMALIZATIONS = {"mean", "paper"}


class WHCCompactDT1DAdapter(DT1DAdapter):
    """Weighted Hartley-cosine compact DT1D adapter for BCHW feature maps."""

    method_name = "WHC-Compact-DT1D"
    proposal_name = "WHC-Compact-DT1D-R124-P2-L1-FixedGate001"
    fixed_final_offsets = (1, 2, 4)

    def __init__(
        self,
        C: int,
        *,
        axis: str = "hw",
        group_size: int = 16,
        residual_scale: float = 1.0,
        gate_init: float = 0.01,
        gate_mode: str = "fixed",
        padding_mode: str = "replicate",
        contrast_split: int = 8,
        detail_basis: str = "orth",
        detail_components: str = "offset4",
        active_offsets: str | Sequence[int] | None = "1,2,4",
        use_pointwise: bool = False,
        pointwise_ratio: int = 32,
        pointwise_groups: int = 4,
        use_bn: bool = False,
        cache_kernel: bool = False,
        project_l1: bool = True,
        whc_p: int = 2,
        whc_lambda_mode: str = "learned",
        whc_lambda_scope: str = "block",
        whc_lambda_init: float = 0.0,
        whc_lambda_max: float = 0.5,
        whc_shift_normalization: str = "mean",
    ) -> None:
        detail_components = str(detail_components).lower()
        if detail_components not in {"offset4", "none"}:
            raise ValueError(
                "WHC-Compact-DT1D supports detail_components='offset4' "
                "(final) or 'none' (ablation)."
            )

        offsets = _parse_offsets(active_offsets)
        p = int(whc_p)
        if p <= 0:
            raise ValueError(f"whc_p must be a positive integer, got {whc_p!r}")

        lambda_mode = str(whc_lambda_mode).lower()
        lambda_scope = str(whc_lambda_scope).lower()
        shift_norm = str(whc_shift_normalization).lower()
        if lambda_mode not in _LAMBDA_MODES:
            raise ValueError(f"whc_lambda_mode must be one of {_LAMBDA_MODES}")
        if lambda_scope not in _LAMBDA_SCOPES:
            raise ValueError(f"whc_lambda_scope must be one of {_LAMBDA_SCOPES}")
        if shift_norm not in _SHIFT_NORMALIZATIONS:
            raise ValueError(
                f"whc_shift_normalization must be one of {_SHIFT_NORMALIZATIONS}"
            )

        lambda_max = float(whc_lambda_max)
        lambda_init = float(whc_lambda_init)
        if lambda_max <= 0:
            raise ValueError("whc_lambda_max must be > 0")
        if abs(lambda_init) > lambda_max + 1e-12:
            raise ValueError(
                f"|whc_lambda_init| must be <= whc_lambda_max ({lambda_max}), "
                f"got {lambda_init}"
            )
        if lambda_mode == "off" and abs(lambda_init) > 1e-12:
            raise ValueError("whc_lambda_init must be 0 when whc_lambda_mode='off'")

        super().__init__(
            C=C,
            axis=axis,
            group_size=group_size,
            residual_scale=residual_scale,
            gate_init=gate_init,
            gate_mode=gate_mode,
            padding_mode=padding_mode,
            contrast_split=contrast_split,
            detail_basis=detail_basis,
            detail_components=detail_components,
            active_offsets=offsets,
            use_pointwise=use_pointwise,
            pointwise_ratio=pointwise_ratio,
            pointwise_groups=pointwise_groups,
            use_bn=use_bn,
            cache_kernel=cache_kernel,
            # WHC projection is applied after the weighted kernel is fused.
            project_l1=False,
        )

        self.project_l1 = bool(project_l1)
        self.whc_p = p
        self.whc_lambda_mode = lambda_mode
        self.whc_lambda_scope = lambda_scope
        self.whc_lambda_max = lambda_max
        self.whc_shift_normalization = shift_norm

        scalar_count = 1 if lambda_scope == "block" else self.num_axes
        if lambda_mode == "learned":
            ratio = max(-0.999999, min(0.999999, lambda_init / lambda_max))
            theta_init = math.atanh(ratio)
            self.whc_theta = nn.Parameter(torch.full((scalar_count,), theta_init))
        else:
            fixed = lambda_init if lambda_mode == "fixed" else 0.0
            self.register_buffer(
                "whc_lambda_fixed",
                torch.full((scalar_count,), fixed, dtype=torch.float32),
                persistent=True,
            )

        self.base_radius = self._infer_base_radius()
        self.base_kernel_size = 2 * self.base_radius + 1
        # Learned weighting is structurally active even when lambda starts at 0.
        self.weighting_active = not (
            self.whc_lambda_mode == "off"
            or (
                self.whc_lambda_mode == "fixed"
                and abs(lambda_init) <= 1e-12
            )
        )
        self.effective_radius = self.base_radius + (
            self.whc_p if self.weighting_active else 0
        )
        self.effective_kernel_size = 2 * self.effective_radius + 1
        self.is_whc_compact_dt1d_adapter = True
        self.implementation = "whc_compact_dt1d_fused_weighted_axial"

    def _infer_base_radius(self) -> int:
        radius = max(self.active_offsets) if self.active_offsets else 0
        if self.detail_components == "offset4":
            radius = max(radius, 4)
        return int(radius)

    def whc_lambda(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.whc_lambda_mode == "learned":
            value = self.whc_lambda_max * torch.tanh(
                self.whc_theta.to(device=device, dtype=dtype)
            )
        else:
            value = self.whc_lambda_fixed.to(device=device, dtype=dtype)
        if self.whc_lambda_scope == "block":
            value = value.expand(self.num_axes)
        return value

    def _crop_base_kernel(self, full17: torch.Tensor) -> torch.Tensor:
        center = 8
        r = self.base_radius
        cropped = full17[..., center-r:center+r+1]
        if cropped.shape[-1] != self.base_kernel_size:
            raise RuntimeError(f"Unexpected base kernel shape: {tuple(cropped.shape)}")
        return cropped

    @staticmethod
    def _shift_pair(base: torch.Tensor, p: int, scale: float) -> torch.Tensor:
        k = int(base.shape[-1])
        out = base.new_zeros(*base.shape[:-1], k + 2 * p)
        out[..., :k] += base
        out[..., 2*p:2*p+k] += base
        return out * float(scale)

    def build_unprojected_kernels(
        self, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        full17 = super().build_kernels(device, dtype, project=False).squeeze(2)
        base = self._crop_base_kernel(full17)
        if not self.weighting_active:
            return base.unsqueeze(2)

        p = self.whc_p
        centered = F.pad(base, (p, p))
        shift_scale = 0.5 if self.whc_shift_normalization == "mean" else 1.0
        weighted = self._shift_pair(base, p, shift_scale)
        lam = self.whc_lambda(device, dtype).view(self.num_axes, 1, 1)
        mixed = (1.0 - lam) * centered + lam * weighted
        return mixed.unsqueeze(2)

    def build_kernels(
        self,
        device: torch.device,
        dtype: torch.dtype,
        *,
        project: bool | None = None,
    ) -> torch.Tensor:
        if project is None:
            project = self.project_l1
        kernel = self.build_unprojected_kernels(device, dtype)
        if project:
            joint_l1 = kernel.abs().sum(dim=-1).sum(dim=0).squeeze(-1)
            scale = torch.maximum(joint_l1, torch.ones_like(joint_l1))
            kernel = kernel / scale.view(1, self.C, 1, 1)
        if int(kernel.shape[-1]) != self.effective_kernel_size:
            raise RuntimeError(
                f"Expected K{self.effective_kernel_size}, got {tuple(kernel.shape)}"
            )
        return kernel

    def parameter_count_breakdown(self) -> Dict[str, int]:
        result = super().parameter_count_breakdown()
        whc_weight = self.whc_theta.numel() if hasattr(self, "whc_theta") else 0
        result["whc_weight"] = int(whc_weight)
        result["total"] = int(result["total"] + whc_weight)
        result["base_kernel_size"] = int(self.base_kernel_size)
        result["effective_kernel_size"] = int(self.effective_kernel_size)
        return result

    @property
    def convolution_calls_per_forward(self) -> int:
        return self.num_axes

    def extra_repr(self) -> str:
        return (
            f"C={self.C}, axis={self.axis}, group={self.group_size}, "
            f"offsets={self.active_offsets}, baseK={self.base_kernel_size}, "
            f"p={self.whc_p}, effectiveK={self.effective_kernel_size}, "
            f"lambda_mode={self.whc_lambda_mode}, scope={self.whc_lambda_scope}, "
            f"lambda_max={self.whc_lambda_max}, "
            f"shift_norm={self.whc_shift_normalization}, "
            f"project_l1={self.project_l1}, detail={self.detail_components}, "
            f"gate_mode={self.gate_mode}, padding={self.padding_mode}"
        )


class WHCCompactDT1DTokenAdapter(nn.Module):
    """Apply final WHC-Compact-DT1D to ViT patch tokens only."""

    def __init__(
        self,
        embed_dim: int,
        grid_size: Tuple[int, int],
        num_prefix_tokens: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.grid_size = (int(grid_size[0]), int(grid_size[1]))
        self.num_prefix_tokens = int(num_prefix_tokens)
        if min(self.grid_size) <= 0:
            raise ValueError(f"Invalid grid_size={grid_size}")
        if self.num_prefix_tokens < 0:
            raise ValueError("num_prefix_tokens must be non-negative")
        self.spatial_adapter = WHCCompactDT1DAdapter(C=self.embed_dim, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"WHCCompactDT1DTokenAdapter expects BND tokens, got {tuple(x.shape)}"
            )
        B, N, D = x.shape
        if D != self.embed_dim:
            raise ValueError(f"Embedding mismatch: got {D}, expected {self.embed_dim}")
        gh, gw = self.grid_size
        expected = self.num_prefix_tokens + gh * gw
        if N != expected:
            raise ValueError(
                f"N={N}, expected {expected} for grid {gh}x{gw} with "
                f"{self.num_prefix_tokens} prefix token(s)"
            )
        prefix = x[:, :self.num_prefix_tokens, :]
        patches = x[:, self.num_prefix_tokens:, :]
        fmap = patches.transpose(1, 2).reshape(B, D, gh, gw)
        fmap = self.spatial_adapter(fmap)
        patches = fmap.reshape(B, D, gh * gw).transpose(1, 2)
        return torch.cat((prefix, patches), dim=1) if self.num_prefix_tokens else patches

    def parameter_count_breakdown(self) -> Dict[str, int]:
        return self.spatial_adapter.parameter_count_breakdown()


__all__ = ["WHCCompactDT1DAdapter", "WHCCompactDT1DTokenAdapter"]
