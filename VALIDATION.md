# Validation

The cleaned ViT integration was checked against the canonical final DT1D implementation.

- Final DT1D spatial kernels: exact numerical equivalence under matched parameters (max absolute difference 0.0).
- Final DT1D forward output: numerical equivalence under matched parameters.
- Unit tests: `4 passed` for final defaults, joint L1 projection, ViT class-token preservation/backpropagation, and zero-mean group modulation.
- Three-seed protocol: fixed paper seeds `0,1,2` with seed-specific output directories.
- Model selection: validation accuracy selects the best epoch; test evaluation is performed once after restoring best-validation trainable parameters.
- Legacy proposal scan: no HCC/shifted-routing/multi-dilation proposal files or active config keys remain.

Full ViT training requires the dependencies in `requirements.txt`; VTAB additionally requires `requirements-vtab.txt` and prepared VTAB data.
