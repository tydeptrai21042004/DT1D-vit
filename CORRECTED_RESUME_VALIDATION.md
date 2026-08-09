# Corrected clean repository / resume runner validation

Date: 2026-08-09

## Fixed

- DT1D `ACTIVE_OFFSETS` is generated using the exact type required by the checked-out YACS schema (`str`, `tuple`, or `list`).
- Every generated tuning YAML is merged through the real repository `get_cfg()` before GPU work on Kaggle.
- Final YAMLs are also preflight-merged before final three-seed runs.
- Existing `run_summary.json` files are reused only when the complete protocol signature is compatible.
- Compatible completed jobs print `[SKIP]` and are not rerun.
- Incompatible/stale jobs print `[RERUN]`, only that seed directory is removed, and only that job is recomputed.
- VPT implementation hashes remain source-faithful to the supplied original VPT code.
- VPT/Linear keep source-faithful SGD + momentum and LR scaling; Full FT uses AdamW; DT1D/Pfeiffer use AdamW. Equal LR-trial counts are retained.
- VTAB remains train800 -> val200 -> official test at best-validation checkpoint.

## Local tests completed

- Python syntax compilation: 79/79 files passed.
- Shell syntax: 3/3 Kaggle scripts passed `bash -n`.
- Original VPT source hash verification: PASS.
- DT1D forward pass accepts `ACTIVE_OFFSETS` as string/list/tuple: 3/3 passed.
- Schema adaptation helper preserves string/list/tuple type and canonical offsets: 3/3 passed.
- Resume simulation: compatible completed job was skipped.
- Stale/incompatible summary simulation: correctly rejected.
- Cache cleanup: 0 `__pycache__`, 0 `.pyc`, 0 `.pyo` at packaging time.

## Kaggle-specific fail-closed check

The resume Kaggle cell runs `run_fair_vit_comparison.py --dry-run` after installing YACS/fvcore and before dataset/GPU training. That dry-run generates all 50 VTAB tuning YAMLs and merges every YAML through the actual repository config schema. The GPU benchmark starts only after the preflight succeeds.

The current local sandbox does not provide the `yacs`/`fvcore` packages, so the real YACS merge cannot be executed locally here. The Kaggle cell performs that exact check before spending GPU time.
