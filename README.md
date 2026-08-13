# WHC-Compact-DT1D on Vision Transformer

This repository is a cleaned ViT classification runner for the final WHC-Compact-DT1D proposal. The previous DT1D-Adapter remains available only as a reference baseline, alongside VPT, Pfeiffer, linear probing, and full fine-tuning.


## Reviewer-safe fair comparison protocol

For cross-method paper results, use **`run_fair_vit_comparison.py`** instead of hand-written per-method solver settings. The runner enforces the same data split, ViT-B/16 checkpoint, resolution, batch size, epoch budget, weight decay, cosine schedule, tuning seed, final seeds, validation-only selection rule, and **the same number of LR-tuning trials**. Optimizer/LR scale are method-faithful: the original VPT and Linear paths use SGD+momentum with the original batch-scaled LR convention, while Full FT/WHC-Compact-DT1D/previous-DT1D/Pfeiffer use AdamW. The VPT implementation files are hash-checked against the user-supplied original VPT source before any run. Test is disabled during tuning and is evaluated once per final seed after restoring that seed's best-validation checkpoint. See `FAIR_COMPARISON.md` and `VPT_SOURCE_FIDELITY.md`.

Run the fast source/protocol check before expensive GPU jobs:

```bash
python verify_fair_protocol.py
python verify_vpt_original.py --batch-size 32 --tokens 5
```

See [`FAIR_COMPARISON.md`](FAIR_COMPARISON.md) and the ready-to-paste Kaggle scripts in [`kaggle_cells/`](kaggle_cells/).

## Final WHC-Compact-DT1D architecture

The final token adapter reshapes ViT patch tokens to a 2D feature grid, applies the WHC-Compact-DT1D spatial operator, and restores the token sequence while leaving the class token unchanged. The selected configuration is:

- height + width axial filtering;
- group-shared compact symmetric support `{0, ±1, ±2, ±4}` (base kernel K9);
- normalized radius-4 zero-sum channel correction `psi4`;
- learned block-shared WHC interpolation `lambda = 0.5*tanh(theta)`, initialized at zero;
- weighted shift parameter `p = 2`;
- fused effective kernel K13 with no extra convolution branch;
- joint H/W L1 projection after WHC weighting;
- **fixed residual gate `gamma = 0.01`**;
- no pointwise channel mixer.

The fused kernel is

```text
k_W = (1-lambda) k + lambda/2 (S_-2 k + S_+2 k).
```

The previous `DT1D` R1248/learned-gate implementation is retained only for direct reviewer comparison.

## Three-seed single-method runs

`run_three_seeds.py` is retained for single-method checks, but it does **not** perform equal-budget hyperparameter selection across baselines. Do not use it as the source of a cross-method fairness claim. For the manuscript comparison, use `run_fair_vit_comparison.py` above.

```bash
python run_three_seeds.py --config-file configs/finetune/flowers_whc_dt1d.yaml
```

The paper final seeds are exactly `0,1,2`. The trainer selects the checkpoint by validation accuracy and evaluates the test set once after restoring the best-validation trainable parameters.

Aggregate a completed three-seed experiment with:

```bash
python aggregate_three_seeds.py outputs/whc_dt1d_flowers
```

## Final WHC config keys

Use `MODEL.ADAPTER.NAME: "WHC_DT1D"` and configure `MODEL.ADAPTER.WHC_DT1D`. See `configs/finetune/flowers_whc_dt1d.yaml` and `configs/vtab/caltech101_whc_dt1d.yaml`. Reviewer-driven design controls are listed in `configs/ablations/whc_p2_fixed_gate_vit.yaml`.

## Tests

```bash
pytest -q tests/test_dt1d_token_adapter.py tests/test_whc_compact_dt1d_token_adapter.py
```

## Upstream attribution

The training/data/backbone infrastructure is based on Visual Prompt Tuning (Jia et al., ECCV 2022). The VPT and other third-party components retain their original licenses and attribution.

## License

The majority of VPT is licensed under the CC-BY-NC 4.0 license (see [LICENSE](https://github.com/KMnP/vpt/blob/main/LICENSE) for details). Portions of the project are available under separate license terms: GitHub - [google-research/task_adaptation](https://github.com/google-research/task_adaptation) and [huggingface/transformers](https://github.com/huggingface/transformers) are licensed under the Apache 2.0 license; [Swin-Transformer](https://github.com/microsoft/Swin-Transformer), [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) and [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch) are licensed under the MIT license; and [MoCo-v3](https://github.com/facebookresearch/moco-v3) and [MAE](https://github.com/facebookresearch/mae) are licensed under the Attribution-NonCommercial 4.0 International license.
