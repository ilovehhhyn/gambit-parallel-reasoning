#!/usr/bin/env bash
# Builds the Gambit Python environment on the Della login node from the fork and prebuilt vLLM binaries, with no CUDA compile.
#
# Run from a clone of the fork at $GAMBIT_ROOT:
#   ( nohup bash repro/della/setup_env.sh > repro/della/setup.log 2>&1 < /dev/null & )
# Each step is idempotent, so a rerun resumes after the last step that finished.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$GAMBIT_ROOT"

echo "[1/7] virtual environment"
if [[ ! -x "$GAMBIT_PYTHON" ]]; then
  "$UV_BIN" venv --python 3.12 .venv
fi

echo "[2/7] base vLLM wheel for commit $VLLM_BASE_COMMIT"
mkdir -p "$GAMBIT_DEPS"
if [[ ! -f "$VLLM_BASE_WHEEL" ]]; then
  curl -sSL --fail -o "$VLLM_BASE_WHEEL.partial" "$VLLM_BASE_WHEEL_URL"
  mv "$VLLM_BASE_WHEEL.partial" "$VLLM_BASE_WHEEL"
fi
echo "$VLLM_BASE_WHEEL_SHA256  $VLLM_BASE_WHEEL" | sha256sum --check --strict

echo "[3/7] torch and build tools"
"$UV_BIN" pip install --python "$GAMBIT_PYTHON" "${TORCH_PINS[@]}"
"$UV_BIN" pip install --python "$GAMBIT_PYTHON" -r requirements/build.txt

echo "[4/7] editable fork with the base binaries copied in"
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION="$VLLM_BASE_WHEEL" \
  "$UV_BIN" pip install --python "$GAMBIT_PYTHON" --no-build-isolation -e .

echo "[5/7] evaluation dependencies"
"$UV_BIN" pip install --python "$GAMBIT_PYTHON" -r gambit/requirements.txt \
  --override repro/della/overrides.txt

echo "[6/7] model $MODEL_ID at $MODEL_REVISION"
"$GAMBIT_PYTHON" -c 'import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision="main"))' "$MODEL_ID"
model_ref="$HF_HOME/hub/models--${MODEL_ID//\//--}/refs/main"
resolved_revision="$(cat "$model_ref")"
if [[ "$resolved_revision" != "$MODEL_REVISION" ]]; then
  echo "model revision mismatch: main resolves to $resolved_revision, pinned $MODEL_REVISION; update MODEL_REVISION in env.sh after reviewing the change" >&2
  exit 1
fi

echo "[7/7] import check"
"$GAMBIT_PYTHON" - "$GAMBIT_ROOT" <<'PY'
import sys
from pathlib import Path

import torch
import vllm
from vllm import _C  # noqa: F401  # loads the prebuilt kernels against this torch build
from vllm.config.gambit import GambitConfig

root = Path(sys.argv[1]).resolve()
package = Path(vllm.__file__).resolve()
if root not in package.parents:
    raise SystemExit(f"vllm imports from {package}, not from the fork at {root}; reinstall the fork in editable mode")
print("vllm", vllm.__version__, "from", package.parent)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gambit defaults", GambitConfig())
PY
echo "SETUP_DONE"
