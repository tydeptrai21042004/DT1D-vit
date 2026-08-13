#!/usr/bin/env python3
"""Dependency-light structural validation for final WHC-Compact-DT1D on ViT-B/16."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.models.vit_adapter.dt1d_adapter import DT1DTokenAdapter
from src.models.vit_adapter.whc_compact_dt1d_adapter import WHCCompactDT1DTokenAdapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    dim, grid, blocks = 768, (14, 14), 12
    old = DT1DTokenAdapter(dim, grid)
    new = WHCCompactDT1DTokenAdapter(dim, grid)
    m = new.spatial_adapter

    old_per = sum(p.numel() for p in old.parameters())
    new_per = sum(p.numel() for p in new.parameters())
    old_all = old_per * blocks
    new_all = new_per * blocks

    h, w = grid
    old_macs = blocks * 2 * dim * h * w * 17
    new_macs = blocks * 2 * dim * h * w * m.effective_kernel_size

    # Stability and trainability checks.
    torch.manual_seed(0)
    with torch.no_grad():
        m.base_coefficients.normal_(0, 0.1)
        m.detail_coefficients.normal_(0, 0.1)
    k = m.build_kernels(torch.device("cpu"), torch.float32)
    joint = k.squeeze(2).abs().sum(-1).sum(0)

    x = torch.randn(2, 197, dim, requires_grad=True)
    y = new(x)
    loss = y.square().mean()
    loss.backward()

    result = {
        "proposal": m.proposal_name,
        "base_support": list(m.active_offsets),
        "whc_p": m.whc_p,
        "base_kernel_size": m.base_kernel_size,
        "effective_kernel_size": m.effective_kernel_size,
        "gate_mode": m.gate_mode,
        "gate_value": float(m.gate.detach()),
        "gate_is_trainable": isinstance(m.gate, torch.nn.Parameter),
        "lambda_mode": m.whc_lambda_mode,
        "lambda_scope": m.whc_lambda_scope,
        "lambda_init": float(m.whc_lambda(torch.device('cpu'), torch.float32)[0].detach()),
        "joint_l1_max": float(joint.max().detach()),
        "lambda_gradient_abs_sum": float(m.whc_theta.grad.abs().sum()),
        "class_token_preserved": bool(torch.allclose(y[:, :1], x[:, :1])),
        "vit_b16_blocks": blocks,
        "old_dt1d_adapter_params_per_block": old_per,
        "whc_adapter_params_per_block": new_per,
        "old_dt1d_adapter_params_12_blocks": old_all,
        "whc_adapter_params_12_blocks": new_all,
        "adapter_param_reduction_pct": 100.0 * (old_all - new_all) / old_all,
        "old_dt1d_axial_macs_224": old_macs,
        "whc_axial_macs_224": new_macs,
        "adapter_mac_reduction_pct": 100.0 * (old_macs - new_macs) / old_macs,
    }

    assert result["proposal"].endswith("FixedGate001")
    assert result["base_support"] == [1, 2, 4]
    assert result["whc_p"] == 2
    assert result["effective_kernel_size"] == 13
    assert result["gate_mode"] == "fixed"
    assert abs(result["gate_value"] - 0.01) < 1e-7
    assert result["gate_is_trainable"] is False
    assert result["joint_l1_max"] <= 1.000001
    assert result["lambda_gradient_abs_sum"] > 0
    assert result["class_token_preserved"]

    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
