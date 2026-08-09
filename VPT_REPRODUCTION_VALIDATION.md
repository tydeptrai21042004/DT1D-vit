# VPT reproduction validation

Validation basis: user-supplied `vpt-main.zip`.

## Result

**PASS** — the corrected repository uses the original VPT model/build implementation and restores the original baseline configuration files used by VPT/Linear/Full FT. The paper comparison then applies a controlled, equal-budget evaluation wrapper around those methods.

## Files verified byte-identical to the supplied source

- `src/models/vit_prompt/vit.py`
- `src/models/vit_prompt/vit_ablations.py`
- `src/models/build_vit_backbone.py`
- `src/models/build_model.py`
- `configs/base-prompt.yaml`
- `configs/prompt/flowers.yaml`
- `configs/base-linear.yaml`
- `configs/base-finetune.yaml`

The fair runner aborts if any of these hashes drift.

## VPT behavior preserved

- shallow prompt tuning;
- prepend prompt placement;
- random prompt initialization;
- no prompt projection (`PROJECT=-1`);
- original CLS pooling;
- prompt dropout 0;
- SGD optimizer;
- momentum 0.9;
- source-default weight decay `1e-4`;
- original tuning convention `effective_lr = nominal_lr * batch_size / 256`.

For batch size 32, the VPT effective LR candidates are:

`0.00625, 0.0125, 0.03125, 0.0625, 0.125, 0.3125, 0.625, 1.25, 3.125, 6.25`.

For batch size 16:

`0.003125, 0.00625, 0.015625, 0.03125, 0.0625, 0.15625, 0.3125, 0.625, 1.5625, 3.125`.

This fixes the previous invalid setup where VPT was run with SGD and LR `1e-3`, which is far below the source tuning range for the batch sizes used in the manuscript.

## Prompt-token policy

- Flowers102: 5 tokens, matching the supplied source default and the original Flowers prompt config.
- VTAB-Caltech101: 10 tokens, preserving the manuscript's existing ~86,118-trainable-parameter VPT setting. The supplied source ZIP does not include the separately distributed dataset-specific VTAB prompt-length hyperparameter file, so the 10-token value is explicitly treated as manuscript-defined rather than source-derived.

## Fair comparison policy

All methods use the same:

- dataset/split;
- ViT-B/16 pretrained checkpoint;
- image resolution;
- batch size;
- epoch budget;
- cosine scheduler;
- fixed weight decay;
- tuning seed;
- final seeds 0/1/2;
- number of LR tuning trials (10 per method by default);
- validation-only hyperparameter selection;
- best-validation checkpoint rule;
- final test policy;
- seeded classifier head and DataLoader stream.

Optimizer family and LR scale are method-native, because forcing the same optimizer/LR on structurally different methods is not a faithful baseline comparison.

## Controlled deviations from the original VPT training scripts

The paper protocol intentionally differs from the full original VPT recipe in these ways:

- the manuscript uses a common 10-epoch budget instead of the original 100-epoch FGVC base schedule;
- warmup is 1 epoch for that 10-epoch budget;
- weight decay is fixed at `1e-4` for every method instead of giving only selected baselines a larger WD search;
- VTAB train800 and val200 stay separated; the original VPT final stage retrains on train800+val200, while the reviewer-facing protocol reports test results from validation-selected checkpoints under one common rule;
- final seeds are 0,1,2.

Therefore, results from `run_fair_vit_comparison.py` should be described as a **controlled, source-faithful VPT baseline under the manuscript's common training budget**, not as an exact reproduction of every training detail in the original VPT paper.

## Validation executed

- `python verify_fair_protocol.py`: PASS
- `python verify_vpt_original.py --batch-size 32 --tokens 5`: PASS
- Flowers102 fair-runner dry run: PASS
- VTAB-Caltech101 fair-runner dry run: PASS
- equal 10-candidate LR budget for every method: PASS
- all 79 Python files compile: PASS
- both Kaggle runner shell scripts pass `bash -n`: PASS
