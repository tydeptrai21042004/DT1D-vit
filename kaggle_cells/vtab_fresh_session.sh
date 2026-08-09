#!/usr/bin/env bash
set -Eeuo pipefail

# VTAB-Caltech101 / ViT-B/16 fair comparison, fresh Kaggle session.
# First invocation in a fresh session runs all missing jobs.
# Re-running this cell in the SAME Kaggle session skips compatible completed jobs.

WORKDIR="${WORKDIR:-/kaggle/working}"
DATA_DIR="${DATA_DIR:-$WORKDIR/vtab_data}"
MODEL_ROOT="${MODEL_ROOT:-$WORKDIR/vit_weights}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$WORKDIR/vtab_caltech101_vitb16_fair_v3}"
RESULT_ZIP="${RESULT_ZIP:-$WORKDIR/vtab_caltech101_vitb16_fair_v3.zip}"
PREFLIGHT_ROOT="${PREFLIGHT_ROOT:-$WORKDIR/vtab_caltech101_preflight_v3}"

# The script is expected to be executed from the corrected repository root.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "============================================================"
echo "SOURCE"
echo "============================================================"
git rev-parse HEAD 2>/dev/null || echo "local/source archive (no git metadata)"

# Dependencies. Do not replace Kaggle's CUDA-enabled torch/torchvision.
python -m pip install -q --upgrade-strategy only-if-needed \
  scipy scikit-learn pandas Pillow fvcore iopath yacs simplejson termcolor \
  tabulate tqdm ml-collections 'timm>=1.0.0,<2' PyYAML tensorflow-datasets

if ! python - <<'PY'
import tensorflow
print('tensorflow:', tensorflow.__version__)
PY
then
  python -m pip install -q 'tensorflow>=2.16,<2.20'
fi

# Source/protocol checks before downloads or GPU jobs.
python verify_vpt_original.py
python verify_fair_protocol.py

# Real generated-config/YACS preflight. This catches schema errors such as
# ACTIVE_OFFSETS str-vs-tuple BEFORE any expensive training starts.
rm -rf "$PREFLIGHT_ROOT"
mkdir -p "$DATA_DIR" "$MODEL_ROOT"
python run_fair_vit_comparison.py \
  --dataset vtab-caltech101 \
  --data-path "$DATA_DIR" \
  --model-root "$MODEL_ROOT" \
  --output-root "$PREFLIGHT_ROOT" \
  --batch-sizes 32 \
  --epochs 10 \
  --warmup-epoch 1 \
  --vpt-tokens 10 \
  --gpus 0,1 \
  --dry-run
rm -rf "$PREFLIGHT_ROOT"

echo "PRE-FLIGHT COMPLETE. No GPU experiment has run yet."

# ViT-B/16 pretrained checkpoint.
WEIGHT_FILE="$MODEL_ROOT/ViT-B_16-224.npz"
if [[ ! -s "$WEIGHT_FILE" ]]; then
  curl -L --fail --retry 5 --retry-delay 5 \
    'https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16-224.npz' \
    -o "$WEIGHT_FILE"
fi
python - "$WEIGHT_FILE" <<'PY'
import sys, numpy as np
z = np.load(sys.argv[1])
assert z.files, 'invalid ViT checkpoint'
print('ViT checkpoint tensors:', len(z.files))
PY

# VTAB-Caltech101 dataset.
python - "$DATA_DIR" <<'PY'
import sys, tensorflow_datasets as tfds
root = sys.argv[1]
builder = tfds.builder('caltech101:3.*.*', data_dir=root)
builder.download_and_prepare()
print('VTAB Caltech101 prepared:', root)
PY

# IMPORTANT: do not rm -rf OUTPUT_ROOT here.
# run_fair_vit_comparison.py checks run_summary.json and prints:
#   [SKIP] compatible completed run
#   [RERUN] incompatible completed run
#   [RUN] missing run
python run_fair_vit_comparison.py \
  --dataset vtab-caltech101 \
  --data-path "$DATA_DIR" \
  --model-root "$MODEL_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --batch-sizes 32 \
  --epochs 10 \
  --warmup-epoch 1 \
  --vpt-tokens 10 \
  --gpus 0,1

rm -f "$RESULT_ZIP"
cd "$WORKDIR"
zip -qr "$RESULT_ZIP" "$(basename "$OUTPUT_ROOT")"
echo "RESULT ZIP: $RESULT_ZIP"
ls -lh "$RESULT_ZIP"
