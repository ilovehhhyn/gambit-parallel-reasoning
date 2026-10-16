#!/usr/bin/env bash
# Runs a scaled-down Gambit and parallel-sampling job on one GPU to show the whole stack works end to end.
#
# The paper's settings (N=256, warmup 12,000 tokens, up to 60,000 tokens per trace) cost hours per
# problem on an 80 GB card. These smaller settings reach every stage within minutes: hidden-state
# scoring, tournaments that prune and branch, scheduler eviction under a deliberately small KV
# cache, the trace-tree export, the vote and the answer checker.
#
#   mkdir -p /scratch/gpfs/ARORA/hh9077/gambit/repro_results/logs
#   sbatch repro/della/smoke.sbatch
#SBATCH --job-name=gambit-smoke
#SBATCH --account=arora
#SBATCH --qos=gpu-test
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --output=/scratch/gpfs/ARORA/hh9077/gambit/repro_results/logs/%x-%j.out
set -euo pipefail
trap 'kill 0' EXIT

source /scratch/gpfs/ARORA/hh9077/gambit/repro/della/env.sh
export HF_HUB_OFFLINE=1
cd "$GAMBIT_ROOT"

NUM_TRACES=16
SWAP_K=2
CHECK_INTERVAL=50        # decision rounds, as in the engine
WARMUP_TOKENS=1500
HARD_FLOOR=0.1
MAX_TOKENS=10000
MAX_MODEL_LEN=32768      # a branch's prompt is its parent's whole sequence, so allow more than MAX_TOKENS
GPU_MEMORY_UTILIZATION=0.3  # about 13 GB of KV cache, below 16 traces x 10,000 tokens, so eviction happens
PROBLEM_INDEX=0          # AIME-25 problem 1, answer 70: short traces, so the checker sees finished answers
RUN_TIMEOUT=18m
RUN_ROOT=$GAMBIT_RESULTS/smoke-$SLURM_JOB_ID

resolved_revision="$(cat "$HF_HOME/hub/models--${MODEL_ID//\//--}/refs/main")"
if [[ "$resolved_revision" != "$MODEL_REVISION" ]]; then
  echo "model revision mismatch: cache has $resolved_revision, pinned $MODEL_REVISION; rerun repro/della/setup_env.sh" >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
git -C "$GAMBIT_ROOT" log --oneline -1

failures=()
run_method() {
  local method=$1
  shift
  local output_dir=$RUN_ROOT/$method
  mkdir -p "$output_dir"
  echo "=== $method: start $(date -Is)"
  local status=0
  timeout --kill-after=60 "$RUN_TIMEOUT" "$GAMBIT_PYTHON" repro/audited_eval.py \
    --method "$method" --num-traces "$NUM_TRACES" \
    --max-tokens "$MAX_TOKENS" --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --benchmark gambit/datasets/aime_2025.jsonl --problem-indices "$PROBLEM_INDEX" \
    --model-path "$MODEL_ID" --scorer-path "$SCORER_PATH" --output-dir "$output_dir" \
    "$@" || status=$?
  echo "=== $method: end $(date -Is) status=$status"
  if (( status == 124 )); then
    failures+=("$method exceeded $RUN_TIMEOUT")
    return
  fi
  if (( status != 0 )); then
    failures+=("$method exited with status $status")
    return
  fi
  local run_dirs=("$output_dir"/*/)
  if (( ${#run_dirs[@]} != 1 )); then
    failures+=("$method wrote ${#run_dirs[@]} run directories, expected 1")
    return
  fi
  "$GAMBIT_PYTHON" repro/check_smoke.py "${run_dirs[0]}" "$method" "$NUM_TRACES" || failures+=("$method smoke check failed")
}

run_method gambit --swap-k "$SWAP_K" --check-interval "$CHECK_INTERVAL" \
  --warmup-tokens "$WARMUP_TOKENS" --hard-floor "$HARD_FLOOR"
run_method parallel-sampling

if (( ${#failures[@]} > 0 )); then
  printf 'SMOKE_FAILED: %s\n' "${failures[@]}" >&2
  exit 1
fi
echo "SMOKE_ALL_OK results in $RUN_ROOT"
