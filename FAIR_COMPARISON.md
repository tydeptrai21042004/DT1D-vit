# Fair ViT comparison protocol

Use `run_fair_vit_comparison.py` for manuscript/reviewer comparisons.

## Fair does not mean forcing one optimizer on every method

The previous failed runs used a tiny SGD learning rate for VPT/Linear while DT1D used AdamW with a much larger LR. That is not a faithful VPT baseline.

The corrected protocol holds the experimental budget equal while preserving method-native optimization:

- same dataset and train/validation/test split;
- same ViT-B/16 pretrained checkpoint;
- same resolution and batch size;
- same epoch budget and cosine schedule;
- same fixed weight decay (`1e-4` by default);
- same tuning seed;
- exactly the same **number of LR candidates** for every method (10 by default);
- same final seeds `0,1,2`;
- same validation-only hyperparameter-selection rule;
- same classifier-head initialization for a given seed;
- same seeded DataLoader/augmentation stream;
- test disabled during tuning;
- final test evaluated once per seed after restoring that seed's best-validation checkpoint.

Method-specific optimizer/LR scale is allowed because the methods were originally designed with different optimization regimes:

| Method | Optimizer in corrected protocol | LR source |
|---|---|---|
| DT1D-Adapter | AdamW | 10-value adapter grid containing manuscript settings |
| Full fine-tuning | AdamW | VPT-source fine-tuning range, batch-scaled and densified to 10 candidates |
| Linear probing | SGD + momentum 0.9 | original VPT linear-tuning range, scaled by `batch_size/256` |
| VPT | SGD + momentum 0.9 | original VPT prompt-tuning range, scaled by `batch_size/256` |
| Pfeiffer Adapter | AdamW | 10-value adapter grid |

This gives each method the same tuning **budget**, without crippling VPT by forcing AdamW/tiny LR.

## Original VPT fidelity

Before generating any experiment config, the runner verifies SHA-256 hashes of the VPT implementation files against the user-supplied original `vpt-main.zip`. If any VPT implementation file changes, the comparison aborts.

The supplied original VPT code uses shallow prompt tuning with prepend/random initialization by default, SGD with momentum 0.9, and the tuning rule

```text
effective_lr = nominal_lr * batch_size / 256
```

The supplied source default is 5 prompt tokens, and Flowers102 uses that source-default configuration. The separately distributed dataset-specific VTAB prompt-length file is not contained in the supplied ZIP. For VTAB-Caltech101, this repository therefore preserves the manuscript's already documented 10-token VPT budget (~86,118 trainable parameters) and records it as manuscript-defined rather than source-derived.

See `VPT_SOURCE_FIDELITY.md` for the exact source hashes and the intentional protocol differences.

## Test leakage prevention

Tuning configs set `DATA.NO_TEST=True`. The selected LR is based only on validation accuracy. The final three-seed run enables the test set and evaluates it only after restoring the best-validation checkpoint.

For VTAB-Caltech101 the paper protocol is:

```text
train800 -> optimization
val200   -> LR/checkpoint selection
test     -> final evaluation only
```

The original VPT `tune_vtab.py` retrains on `train800+val200` in its final phase. We intentionally do **not** do that here because the reviewer requested a clean validation-selected test protocol shared by all methods.

## Recommended commands

Flowers102:

```bash
python run_fair_vit_comparison.py \
  --dataset flowers102 \
  --data-path /kaggle/working/flowers_download/flowers-102/jpg \
  --model-root /kaggle/working/vit_weights \
  --output-root /kaggle/working/flowers102_vitb16_fair \
  --batch-sizes 32,16 \
  --epochs 10 \
  --warmup-epoch 1 \
  --vpt-tokens 5 \
  --gpus 0,1
```

VTAB-Caltech101:

```bash
python run_fair_vit_comparison.py \
  --dataset vtab-caltech101 \
  --data-path /kaggle/working/vtab_data \
  --model-root /kaggle/working/vit_weights \
  --output-root /kaggle/working/vtab_caltech101_vitb16_fair \
  --batch-sizes 32 \
  --epochs 10 \
  --warmup-epoch 1 \
  --vpt-tokens 10 \
  --gpus 0,1
```

## Outputs to trust

- `fairness_audit.json`: source-fidelity check, optimizer profile, LR grids and equal tuning budget.
- `protocol_configs/`: exact YAML files executed.
- `aggregated/*_fair_three_seed.csv`: final mean ± SD results.
- `aggregated/fair_protocol_manifest.json`: selected LR, parameter counts, per-seed results and config hashes.
- `logs/`: per-run logs.

If the best validation LR is at a grid boundary, final testing aborts. Shift/expand that method's grid while keeping the **same number of candidates** as the other methods, then rerun.
