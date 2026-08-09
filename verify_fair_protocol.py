#!/usr/bin/env python3
"""Fast local verification of the fair/source-faithful ViT protocol."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

import run_fair_vit_comparison as fair
from src.models.mlp import build_seeded_classifier_head


def check_source_contracts(root: Path):
    train = (root / "train.py").read_text()
    get_loaders = train.split("def get_loaders", 1)[1].split("def train", 1)[0]
    assert "construct_train_loader(cfg)" in get_loaders
    assert "construct_trainval_loader(cfg)" not in get_loaders

    trainer = (root / "src/engine/trainer.py").read_text()
    marker = "# Restore the best-validation trainable parameters, then evaluate test once."
    assert marker in trainer
    tail = trainer.split(marker, 1)[1].split("return {", 1)[0]
    assert tail.count('self.eval_classifier(test_loader, "test", True)') == 1

    pfeiffer = (root / "src/models/vit_adapter/vit.py").read_text()
    assert "nn.init.xavier_uniform_(self.adapter_downsample.weight)" in pfeiffer
    assert "nn.init.zeros_(self.adapter_upsample.weight)" in pfeiffer

    tfds = (root / "src/data/datasets/tf_dataset.py").read_text()
    preamble = tfds.split("class TFDataset", 1)[0]
    assert "from ..vtab_datasets import diabetic_retinopathy" not in preamble


def check_vpt_source_fidelity(root: Path):
    result = fair.verify_vpt_source_fidelity(root)
    assert result["status"] == "PASS"
    assert set(result["hashes"]) == set(fair.VPT_SOURCE_HASHES)


def check_head_seed():
    torch.manual_seed(11)
    _ = torch.randn(3)
    a = build_seeded_classifier_head(16, [5], seed=2)
    torch.manual_seed(11)
    _ = torch.randn(1000)
    b = build_seeded_classifier_head(16, [5], seed=2)
    assert torch.equal(a.last_layer.weight, b.last_layer.weight)
    assert torch.equal(a.last_layer.bias, b.last_layer.bias)


def make_args(tmp: Path, dataset="flowers102"):
    kw = dict(
        dataset=dataset,
        data_path=str(tmp / "data"),
        model_root=str(tmp / "weights"),
        output_root=str(tmp / "out"),
        resolution=224,
        num_workers=0,
        weight_decay=1e-4,
        warmup_epoch=1,
        epochs=10,
        patience=20,
        log_every=10,
        vpt_tokens=5,
        pfeiffer_reduction=16,
        dt1d_active_offsets=fair.active_offsets_for_repo_schema(),
        lr_grid=None,
    )
    for m in fair.METHOD_ORDER:
        kw[f"{m}_lr_grid"] = None
    return SimpleNamespace(**kw)


def check_fair_config_generation(tmp: Path):
    args = make_args(tmp)
    bs = 32
    grids = {(bs, m): fair.resolve_method_lr_grid(args, m, bs) for m in fair.METHOD_ORDER}

    # Equal tuning budget, but method-faithful scales/optimizers.
    assert {len(v) for v in grids.values()} == {10}
    assert fair.method_optimizer("vpt") == "sgd"
    assert fair.method_optimizer("linear") == "sgd"
    assert fair.method_optimizer("full") == "adamw"
    assert fair.method_optimizer("dt1d") == "adamw"
    assert fair.method_optimizer("pfeiffer") == "adamw"

    # Exact original VPT batch-scaling rule: nominal_lr * B/256.
    expected_vpt = sorted({x * bs / 256.0 for x in fair.VPT_NOMINAL_LRS})
    assert grids[(bs, "vpt")] == expected_vpt

    configs = {}
    for method in fair.METHOD_ORDER:
        for lr in grids[(bs, method)]:
            configs[(bs, method, lr)] = fair.make_config(args, bs, method, lr, "tune")

    audit = fair.audit_tuning_configs(configs, fair.METHOD_ORDER, grids)
    assert audit["status"] == "PASS"
    assert audit["batches"][str(bs)]["trials_per_method"] == 10

    for method in fair.METHOD_ORDER:
        for lr in grids[(bs, method)]:
            cfg = configs[(bs, method, lr)]
            assert cfg["DATA"]["NO_TEST"] is True
            assert cfg["SOLVER"]["OPTIMIZER"] == fair.method_optimizer(method)
            assert cfg["SOLVER"]["BASE_LR"] == lr
            assert cfg["SOLVER"]["WEIGHT_DECAY"] == 1e-4
            assert cfg["SOLVER"]["TOTAL_EPOCH"] == 10
            assert cfg["SOLVER"]["WARMUP_EPOCH"] == 1

    vpt_cfg = configs[(bs, "vpt", grids[(bs, "vpt")][0])]
    p = vpt_cfg["MODEL"]["PROMPT"]
    assert p["NUM_TOKENS"] == 5
    assert p["LOCATION"] == "prepend"
    assert p["INITIATION"] == "random"
    assert p["PROJECT"] == -1
    assert p["DEEP"] is False
    assert p["VIT_POOL_TYPE"] == "original"
    assert p["DROPOUT"] == 0.0

    # Hyperparameter selection must ignore test accuracy completely.
    lr, val = fair.select_lr([
        (0.01, {"best_val_top1": 0.6, "test_top1": 1.0}),
        (0.1, {"best_val_top1": 0.8, "test_top1": 0.0}),
        (1.0, {"best_val_top1": 0.7, "test_top1": 1.0}),
    ])
    assert lr == 0.1 and val == 0.8


def main():
    root = Path(__file__).resolve().parent
    check_source_contracts(root)
    check_vpt_source_fidelity(root)
    check_head_seed()
    with tempfile.TemporaryDirectory() as td:
        check_fair_config_generation(Path(td))
    print("FAIR + VPT SOURCE-FIDELITY VERIFICATION: PASS")
    print(json.dumps({
        "vpt_source_hashes_match": True,
        "equal_lr_trials_per_method": 10,
        "method_native_optimizers": {
            "VPT": "SGD+momentum (original source)",
            "Linear": "SGD+momentum (original source)",
            "Full fine-tuning": "AdamW (original source)",
            "DT1D-Adapter": "AdamW",
            "Pfeiffer Adapter": "AdamW",
        },
        "vpt_lr_rule": "nominal LR * batch_size / 256 (original source tuning rule)",
        "common_weight_decay": 1e-4,
        "tuning_test_disabled": True,
        "selection_metric": "validation only",
        "final_test_policy": "once after best-validation restore",
        "same_head_init_per_seed": True,
        "vtab_train_val_separated": True,
        "pfeiffer_non_dead_init": True,
    }, indent=2))


if __name__ == "__main__":
    main()
