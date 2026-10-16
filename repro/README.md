# Della reproduction

This directory installs Gambit on Princeton's Della cluster and runs a scaled-down job that shows every stage works.
It does not reproduce the paper's accuracy numbers.

## Install without compiling CUDA

The fork's `csrc/`, `cmake/` and `CMakeLists.txt` are identical to vLLM main commit
`d8140b98337a253f73dedd58c7d1e9b8832a9e3f`. vLLM still publishes the prebuilt wheel for that
commit, so `setup_env.sh` copies its compiled libraries into an editable install of the fork.
This takes about 10 minutes instead of the hours the top-level README warns about.

```bash
git clone https://github.com/ilovehhhyn/gambit-parallel-reasoning /scratch/gpfs/ARORA/hh9077/gambit
cd /scratch/gpfs/ARORA/hh9077/gambit
( nohup bash repro/della/setup_env.sh > repro/della/setup.log 2>&1 < /dev/null & )
```

`env.sh` pins the wheel by SHA-256, PyTorch 2.9.0 (CUDA 12.8), and the model revision.

Two install problems in the original instructions are fixed here:

- `uv pip install vllm` installs today's vLLM, not the version the fork is based on. The fork removed its own `torch` pin, so nothing stops the mismatch.
- `gambit/requirements.txt` pins `antlr4-python3-runtime==4.11`, but `latex2sympy2` needs 4.7.2. With 4.11, `gambit/tests/evaluator.py` fails at import, so `benchmark_eval.py` cannot start. `overrides.txt` installs 4.7.2.

## Smoke job

```bash
mkdir -p repro_results/logs
sbatch repro/della/smoke.sbatch
```

The job uses one A100 80 GB on the `gpu-test` queue; it took 7.6 minutes. It runs Qwen3-4B-Thinking-2507 with the
provided `Qwen3-4B_step_scorer.pt` on AIME-25 problem 1, once with Gambit and once with parallel sampling.

| Setting | Paper | Smoke job |
| --- | --- | --- |
| Traces N and capacity C | 256 | 16 |
| Swap size K | 16 | 2 |
| Check interval (decision rounds) | 200 | 50 |
| Warmup tokens | 12,000 | 1,500 |
| Maximum new tokens | 60,000 | 10,000 |
| GPU memory fraction | 0.9 | 0.3, so the KV cache fills and the scheduler evicts traces |

`audited_eval.py` runs `gambit/tests/benchmark_eval.py` unchanged and writes `question_NNNN.audit.json`
beside each result. It compares the script's token count with vLLM's `vllm:generation_tokens` counter.
`check_smoke.py` then fails the job unless scoring, branching, pruning, the vote and the answer checker all happened.

## Result of job 15701044 (2026-10-16, A100 80 GB PCIe)

| Measure | Gambit | Parallel sampling |
| --- | --- | --- |
| Final answer (truth 70) | 70 | 70 |
| Correct completed traces | 16 of 16 | 16 of 16 |
| Generation time | 136 s | 142 s |
| Tokens counted by `benchmark_eval.py` | 145,823 | 134,346 |
| Tokens counted by vLLM's engine counter | 145,785 | 134,346 |
| Branches, pruned traces, deepest branch | 59, 59, depth 4 | none |
| Prompt tokens served from the prefix cache | 373,072 of 373,696 (99.8%) | 720 of 928 |
| Traces evicted by the scheduler | 0 | 2 preemptions |

The KV cache held 107,104 tokens. Gambit pruned partial traces often enough that the scheduler never
evicted one, so this run does not test how evicted traces are counted. The 38-token gap in Gambit's
row is most likely timing: the counter lives in the front-end process, and Gambit's drain loop can
return before the front end reads the engine's last outputs. This is not verified.
These settings are for checking the setup. One easy problem says nothing about the paper's accuracy or token claims.
