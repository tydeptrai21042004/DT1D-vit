#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# VTAB-Caltech101 / ViT-B/16 fair resume runner
# - clones repository from scratch
# - NEVER deletes OUTPUT_ROOT
# - restores a previous result archive when supplied/found
# - validates all generated YAML with real YACS before GPU work
# - skips compatible completed tuning/final jobs
# - reruns only missing or incompatible jobs
# ============================================================

REPO_URL="${REPO_URL:-https://github.com/tydeptrai21042004/DT1D-vit.git}"
REPO_COMMIT="${REPO_COMMIT:-}"
RESUME_ARCHIVE="${RESUME_ARCHIVE:-}"

WORKDIR="/kaggle/working"
REPO_DIR="$WORKDIR/DT1D-vit-fair-vtab-resume"
DATA_DIR="$WORKDIR/vtab_data"
MODEL_ROOT="$WORKDIR/vit_weights"
OUTPUT_ROOT="$WORKDIR/vtab_caltech101_vitb16_fair"
RESULT_ZIP="$WORKDIR/vtab_caltech101_vitb16_fair.zip"
PREFLIGHT_ROOT="$WORKDIR/vtab_caltech101_preflight"

# 1) Clone code only. Keep experiment outputs untouched.
rm -rf "$REPO_DIR"
git clone --depth 1 "$REPO_URL" "$REPO_DIR"
cd "$REPO_DIR"
if [[ -n "$REPO_COMMIT" ]]; then
  git fetch --depth 1 origin "$REPO_COMMIT"
  git checkout --detach "$REPO_COMMIT"
fi

echo "SOURCE COMMIT: $(git rev-parse HEAD)"

# Fail immediately if GitHub does not yet contain the corrected resume-safe code.
python - <<'PY'
from pathlib import Path
s = Path('run_fair_vit_comparison.py').read_text(encoding='utf-8')
required = [
    'active_offsets_for_repo_schema',
    'preflight_yacs_merge',
    '[SKIP] compatible completed run',
    '[RERUN] incompatible completed run',
]
missing = [x for x in required if x not in s]
if missing:
    raise SystemExit(
        'ERROR: cloned repository is not the corrected resume-safe version. '
        'Push the clean corrected repo first. Missing markers: ' + repr(missing)
    )
print('CORRECTED REPO MARKERS: PASS')
PY

# 2) Dependencies. Do not replace Kaggle CUDA torch/torchvision.
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

python verify_vpt_original.py
python verify_fair_protocol.py

# 3) Real config preflight BEFORE downloads/training.
# This catches the exact ACTIVE_OFFSETS str-vs-tuple YACS error.
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

# 4) Restore previous partial/completed outputs if this is a fresh Kaggle session.
# In the same Kaggle session OUTPUT_ROOT already exists, so nothing is copied.
if [[ ! -d "$OUTPUT_ROOT" ]]; then
  if [[ -z "$RESUME_ARCHIVE" ]]; then
    # Prefer an explicitly uploaded previous result zip from /kaggle/input.
    RESUME_ARCHIVE="$(find /kaggle/input -type f -name 'vtab_caltech101_vitb16_fair.zip' -print -quit 2>/dev/null || true)"
  fi
  if [[ -z "$RESUME_ARCHIVE" && -f "$RESULT_ZIP" ]]; then
    RESUME_ARCHIVE="$RESULT_ZIP"
  fi
  if [[ -n "$RESUME_ARCHIVE" && -f "$RESUME_ARCHIVE" ]]; then
    echo "RESTORING PREVIOUS OUTPUTS: $RESUME_ARCHIVE"
    unzip -q -o "$RESUME_ARCHIVE" -d "$WORKDIR"
  else
    echo "NO PREVIOUS OUTPUT ARCHIVE FOUND. Only missing jobs can be skipped if OUTPUT_ROOT already existed."
  fi
else
  echo "EXISTING OUTPUT_ROOT FOUND: $OUTPUT_ROOT"
  echo "Compatible completed jobs will be skipped automatically."
fi

# 5) Pretrained ViT weights.
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

# 6) VTAB Caltech101 data.
python - "$DATA_DIR" <<'PY'
import sys, tensorflow_datasets as tfds
root = sys.argv[1]
builder = tfds.builder('caltech101:3.*.*', data_dir=root)
builder.download_and_prepare()
print('VTAB Caltech101 prepared:', root)
PY

# 7) Resume-safe fair experiment.
# run_fair_vit_comparison.py checks every existing run_summary.json:
#   compatible -> [SKIP]
#   incompatible -> [RERUN] only that job
#   missing -> [RUN]
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

# 8) Package everything for future resume / manuscript aggregation.
rm -f "$RESULT_ZIP"
cd "$WORKDIR"
zip -qr "$RESULT_ZIP" "$(basename "$OUTPUT_ROOT")"
echo "RESULT ZIP: $RESULT_ZIP"
ls -lh "$RESULT_ZIP"
