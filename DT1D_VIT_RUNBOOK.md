# DT1D ViT paper runbook

## Flowers102 / ViT-B/16

```bash
python run_three_seeds.py --config-file configs/finetune/flowers_dt1d.yaml
```

## VTAB-Caltech101 / ViT-B/16

```bash
python run_three_seeds.py --config-file configs/vtab/caltech101_dt1d.yaml
```

The three paper seeds are 0, 1, and 2. Test accuracy is evaluated only after restoring the trainable parameters from the best validation epoch.
