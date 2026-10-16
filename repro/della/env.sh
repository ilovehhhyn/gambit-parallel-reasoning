# Defines the pinned versions, paths and cache locations that every Gambit reproduction script on Della shares.
# Source it; do not execute it.

GAMBIT_ROOT=/scratch/gpfs/ARORA/hh9077/gambit
GAMBIT_PYTHON=$GAMBIT_ROOT/.venv/bin/python
GAMBIT_DEPS=/scratch/gpfs/ARORA/hh9077/gambit_deps
GAMBIT_RESULTS=$GAMBIT_ROOT/repro_results
UV_BIN=$HOME/.local/bin/uv

# The fork's kernel sources (csrc/, cmake/, CMakeLists.txt) are identical to this vLLM main
# commit, so its prebuilt binaries replace the multi-hour CUDA compile.
VLLM_BASE_COMMIT=d8140b98337a253f73dedd58c7d1e9b8832a9e3f
VLLM_BASE_WHEEL=$GAMBIT_DEPS/vllm-0.11.1rc7.dev89+gd8140b983-cp38-abi3-manylinux1_x86_64.whl
VLLM_BASE_WHEEL_URL=https://wheels.vllm.ai/$VLLM_BASE_COMMIT/vllm-0.11.1rc7.dev89%2Bgd8140b983-cp38-abi3-manylinux1_x86_64.whl
VLLM_BASE_WHEEL_SHA256=b3b3ca5a8f3cc771db46e8b2501b8614a9ec9bd06c750a705785a918be638dd9
# The base commit's requirements/cuda.txt pins these; the fork removed them from its own list.
TORCH_PINS=(torch==2.9.0 torchaudio==2.9.0 torchvision==0.24.0)

MODEL_ID=Qwen/Qwen3-4B-Thinking-2507
MODEL_REVISION=768f209d9ea81521153ed38c47d515654e938aea
SCORER_PATH=$GAMBIT_ROOT/gambit/step_scorer_checkpoint/Qwen3-4B_step_scorer.pt

export HF_HOME=/scratch/gpfs/ARORA/hh9077/hf_cache
export UV_CACHE_DIR=/scratch/gpfs/ARORA/hh9077/uv/cache
export UV_PYTHON_INSTALL_DIR=/scratch/gpfs/ARORA/hh9077/uv/python
export VLLM_CACHE_ROOT=$GAMBIT_ROOT/.cache/vllm
export TRITON_CACHE_DIR=$GAMBIT_ROOT/.cache/triton
export FLASHINFER_WORKSPACE_BASE=$GAMBIT_ROOT/.cache/flashinfer
export XDG_CACHE_HOME=$GAMBIT_ROOT/.cache/xdg
# Compute nodes have no CUDA toolkit, so FlashInfer cannot JIT-compile its sampler there
# (fatal error: curand.h). vLLM then uses its PyTorch top-k/top-p sampler.
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
