# Kaggle v4 validation

Root cause confirmed: YACS decodes YAML string scalars with Python literal parsing. The YAML value `1,2,4,8` is interpreted as the tuple `(1, 2, 4, 8)`, so serializing `ACTIVE_OFFSETS` can cause a string-vs-tuple schema mismatch.

v4 fix:
- Generated fair-comparison YAMLs never serialize `MODEL.ADAPTER.DT1D.ACTIVE_OFFSETS`.
- The runner validates that the repository default is semantically `(1, 2, 4, 8)` and inherits that default.
- Shipped DT1D YAML configs also omit the key.
- The real Kaggle pre-flight still merges every generated YAML through `get_cfg().merge_from_file(...)` before any GPU training.

Local checks performed on the packaged source:
- Python compileall: PASS.
- All Kaggle shell scripts `bash -n`: PASS.
- Exact Python literal trap `ast.literal_eval('1,2,4,8') == (1,2,4,8)`: PASS.
- Generated VTAB tuning configs: 50/50 generated; all 10 DT1D configs omit `ACTIVE_OFFSETS`.
- Static DT1D YAML overrides of `ACTIVE_OFFSETS`: 0.
- Cache files after cleanup: 0.

The final authoritative environment check remains the Kaggle YACS pre-flight, which must print `PRE-FLIGHT YACS MERGE: PASS (50/50 configs) [tuning]` before training starts.
