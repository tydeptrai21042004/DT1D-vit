# Fair protocol validation

Validated after comparing this repository against the user-supplied original `vpt-main.zip`.

## Correctness checks

1. **VPT implementation fidelity:** the four VPT model/build files used by the baseline match the original source ZIP byte-for-byte by SHA-256.
2. **VPT optimizer fidelity:** VPT uses SGD with momentum 0.9, as in the original source.
3. **VPT LR fidelity:** the VPT tuning grid uses the original nominal-LR convention and applies `effective_lr = nominal_lr * batch_size / 256`.
4. **VPT structure fidelity:** shallow/prepend/random prompts, no projection, original CLS pooling, dropout 0, and source-default prompt length 5 unless explicitly overridden.
5. **Equal tuning budget:** every compared method receives exactly 10 LR candidates per batch-size setting by default.
6. **Shared protocol:** dataset/splits, pretrained backbone, resolution, batch size, epoch count, cosine scheduler, fixed WD, tuning seed, final seeds, checkpoint rule, and test policy are common across methods.
7. **No test leakage:** tuning runs set `DATA.NO_TEST=True` and hyperparameters are selected only from validation accuracy.
8. **Final test policy:** final seeds are 0/1/2; test is evaluated once after restoring each seed's best-validation trainable state.
9. **VTAB split isolation:** train800 is used for optimization, val200 for selection, official test for final evaluation; no train800+val200 leakage is used in the paper protocol.
10. **Pfeiffer initialization:** down projection is Xavier initialized and up projection is zero initialized, avoiding a dead all-zero adapter branch.
11. **Deterministic comparison:** classifier-head initialization and DataLoader/worker RNG are seed controlled across methods.
12. **Resume protection:** run summaries record optimizer, momentum, LR, WD, architecture signature, dataset and protocol; stale incompatible runs are rejected.
13. **Search-boundary guard:** if a method selects the edge of its LR grid, final testing stops. Its grid must be shifted/expanded while preserving the same candidate count as every other method.
14. **Python syntax:** all repository Python files compile successfully with `py_compile` in the validation environment.
15. **Kaggle runner shell syntax:** both Flowers102 and VTAB-Caltech101 fair runners pass `bash -n`.

## Important interpretation

The corrected paper comparison is **method-faithful and budget-fair**, not "same optimizer for all methods." Forcing VPT to AdamW or to the small LR used by DT1D is specifically disallowed because it does not reproduce the supplied original VPT training behavior.

The paper's 10-epoch controlled comparison also should not be described as an exact reproduction of the original VPT paper's complete training schedule. See `VPT_SOURCE_FIDELITY.md`.
