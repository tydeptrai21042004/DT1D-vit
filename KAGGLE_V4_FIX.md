# Kaggle v4 ACTIVE_OFFSETS fix

Root cause: YACS literal-decodes YAML strings with `ast.literal_eval`. The scalar `1,2,4,8` therefore becomes tuple `(1, 2, 4, 8)` during merge, which conflicts with a string-typed repository default.

Fix: fair comparison configs do not emit `MODEL.ADAPTER.DT1D.ACTIVE_OFFSETS`. The runner validates the repository default is semantically `(1,2,4,8)` and inherits it. All generated tuning and final configs still go through the real YACS pre-flight on Kaggle before training.
