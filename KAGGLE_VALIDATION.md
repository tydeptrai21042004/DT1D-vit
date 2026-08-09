# Kaggle runtime validation (2026-08-08)

This repository revision was validated specifically against the two reviewer/Kaggle cells for Flowers102 and VTAB-Caltech101.

## Fixes included

- Modern `timm.layers` imports with fallback to legacy `timm.models.layers`.
- `MODEL.ADAPTER.DT1D.ACTIVE_OFFSETS` is intentionally not emitted by fair-runner YAMLs. The runner validates the repository default is semantically `(1, 2, 4, 8)` and inherits it, avoiding YACS string literal-decoding into a tuple.
- Canonical Pfeiffer `MODEL.ADAPTER.REDUCTION_FACTOR` is accepted and preferred, while the legacy misspelling `REDUCATION_FACTOR` remains as a compatibility alias.
- AdamW reads `SOLVER.WEIGHT_DECAY` instead of hard-coding `0.01` and uses current PyTorch in-place operation signatures.
- The source blocks patched by the existing Kaggle cells (VTAB train split and Pfeiffer initialization) are intentionally kept patch-compatible.

## Validation performed

- `python -m compileall -q .`: PASS.
- `pytest -q tests/test_dt1d_token_adapter.py`: 8 passed using a modern-timm API compatibility harness.
- Both corrected cell scripts: `bash -n` PASS.
- Flowers Bash array resolves to two elements exactly: `32`, `16`.
- Exact cell patch simulation against this source: PASS for Flowers Pfeiffer patch and VTAB split + Pfeiffer patches.
- Generated YAML schema validation for all five methods (`dt1d`, `full`, `linear`, `vpt`, `pfeiffer`) in both cells: PASS.
- Real repository ViT-B/16 forward/backward on PyTorch 2.10.0 / torchvision 0.25.0:
  - DT1D: PASS, 85,362 trainable parameters.
  - Pfeiffer: PASS, 972,966 trainable parameters, reduction factor 16.
  - VPT: PASS, 86,118 trainable parameters.
  - Linear: PASS, 78,438 trainable parameters.
  - Full fine-tuning: PASS, 85,877,094 trainable parameters.
- Optimizer step:
  - DT1D, Pfeiffer, VPT, Linear: PASS.
  - Full fine-tuning AdamW construction: PASS.
  - AdamW non-bias group uses `weight_decay=0.0001`: PASS.
- Pretrained ViT-B/16 `load_from()` path with shape-correct NPZ-style tensors:
  - Plain ViT (Full/Linear): PASS.
  - Prompted ViT (VPT): PASS.
  - Adapter ViT (DT1D/Pfeiffer): PASS.
- Flowers JSON train/val/test loader contract: PASS with 224x224 transformed mini-batches.
- Trainer summary contract (`best_epoch`, `best_val_top1`, `test_top1`): PASS.
- `aggregate_three_seeds.py`: PASS using seed0/1/2 synthetic summaries.
- Flowers combined CSV: PASS, 10 rows (2 batch sizes x 5 methods).
- VTAB combined CSV: PASS, 5 rows.

## External-runtime boundary

The local validation environment uses PyTorch 2.10.0 + torchvision 0.25.0 CPU, matching the versions shown by the supplied Kaggle logs except for CUDA. It cannot execute Kaggle's external network/GPU runtime directly. The supplied Kaggle logs already demonstrate that the weight download, Flowers download, TensorFlow 2.19/TFDS setup, and Caltech101 download/prepare stages succeed on Kaggle. The failures observed after those stages are addressed by this revision and the corrected Flowers Bash array syntax.
