"""Runs gambit/tests/benchmark_eval.py for one method and records vLLM's engine counters beside each question.

benchmark_eval.py counts a Gambit question's tokens from the trace-tree export. A trace that the
scheduler evicts under KV-cache pressure (a "ghost", paper Section 4.2) is never finalised, so
its tokens never reach that export. vLLM's `vllm:generation_tokens` counter is incremented for
every sampled token, branches included, so per question

    unaccounted_tokens = delta(vllm:generation_tokens) - total_tokens

is the number of generated tokens the script leaves out. For parallel sampling it must be 0.
"""

import argparse
import enum
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gambit" / "tests"))

import benchmark_eval  # noqa: E402  # needs gambit/tests on sys.path for its `evaluator` import

# Paper Section 5.1 values; the check interval counts decision rounds, not tokens.
PAPER_SWAP_K = 16
PAPER_CHECK_INTERVAL = 200
PAPER_WARMUP_TOKENS = 12_000
PAPER_HARD_FLOOR = 0.1
PAPER_MAX_TOKENS = 60_000
PAPER_GPU_MEMORY_UTILIZATION = 0.9


class Method(enum.Enum):
    """The three inference methods the paper compares that this fork can run."""

    PARALLEL_SAMPLING = "parallel-sampling"
    STEP = "step"
    GAMBIT = "gambit"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", required=True, choices=[method.value for method in Method])
    parser.add_argument("--num-traces", type=int, required=True, help="N; Gambit also uses it as capacity C")
    parser.add_argument("--swap-k", type=int, help=f"Gambit only; default {PAPER_SWAP_K}")
    parser.add_argument("--check-interval", type=int, help=f"Gambit only; default {PAPER_CHECK_INTERVAL}")
    parser.add_argument("--warmup-tokens", type=int, help=f"Gambit only; default {PAPER_WARMUP_TOKENS}")
    parser.add_argument("--hard-floor", type=float, help=f"Gambit only; default {PAPER_HARD_FLOOR}")
    parser.add_argument("--max-tokens", type=int, default=PAPER_MAX_TOKENS)
    parser.add_argument("--max-model-len", type=int, default=0, help="0 keeps the model default")
    parser.add_argument("--gpu-memory-utilization", type=float, default=PAPER_GPU_MEMORY_UTILIZATION)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--scorer-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--problem-indices", default="")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def benchmark_eval_argv(args: argparse.Namespace) -> list[str]:
    """Translate one method into the exact benchmark_eval.py command line."""
    method = Method(args.method)
    tournament = [args.swap_k, args.check_interval, args.warmup_tokens, args.hard_floor]
    if method is not Method.GAMBIT and any(value is not None for value in tournament):
        raise SystemExit(f"tournament options apply only to --method gambit, not `{method.value}`; remove them")
    common = [
        "--benchmark", args.benchmark,
        "--output-dir", args.output_dir,
        "--run-label", method.value,
        "--model-path", args.model_path,
        "--gambit-step-scorer-path", args.scorer_path,
        "--num-traces", str(args.num_traces),
        "--max-tokens", str(args.max_tokens),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--seed", str(args.seed),
        "--no-score-history",
    ]
    if args.problem_indices != "":
        common += ["--problem-indices", args.problem_indices]
    if method is Method.PARALLEL_SAMPLING:
        return common
    if method is Method.STEP:
        return common + ["--enable-gambit"]
    if method is Method.GAMBIT:
        swap_k = PAPER_SWAP_K if args.swap_k is None else args.swap_k
        if swap_k > args.num_traces // 2:
            raise SystemExit(f"--swap-k {swap_k} exceeds half of --num-traces {args.num_traces}; the engine would clamp it silently")
        return common + [
            "--enable-gambit", "--enable-branching",
            "--tournament-capacity", str(args.num_traces),
            "--tournament-swap-k", str(swap_k),
            "--tournament-check-interval", str(PAPER_CHECK_INTERVAL if args.check_interval is None else args.check_interval),
            "--tournament-warmup-tokens", str(PAPER_WARMUP_TOKENS if args.warmup_tokens is None else args.warmup_tokens),
            "--tournament-hard-floor", str(PAPER_HARD_FLOOR if args.hard_floor is None else args.hard_floor),
            "--stop-after-completed-traces", str(args.num_traces),
        ]
    raise AssertionError(f"unhandled method {method}")


def read_counters(llm) -> dict[str, float]:
    """Return every Prometheus counter the engine exposes, keyed by name and sorted labels."""
    from vllm.v1.metrics.reader import Counter

    counters: dict[str, float] = {}
    for metric in llm.get_metrics():
        if not isinstance(metric, Counter):
            continue
        labels = ",".join(f"{key}={metric.labels[key]}" for key in sorted(metric.labels))
        counters[f"{metric.name}{{{labels}}}"] = float(metric.value)
    return dict(sorted(counters.items()))


def generation_tokens(counters: dict[str, float]) -> float:
    matching = [value for key, value in counters.items() if key.startswith("vllm:generation_tokens{")]
    if len(matching) != 1:
        raise RuntimeError(f"expected exactly one vllm:generation_tokens counter, found {len(matching)}")
    return matching[0]


def summarize_tree(tree: dict | None) -> dict[str, int] | None:
    """Count trace states in a Gambit tree export, including ghosts whose tokens were never recorded."""
    if tree is None:
        return None
    nodes = list(tree["nodes"].values())
    return {
        "nodes": len(nodes),
        "roots": sum(1 for node in nodes if node["parent_id"] is None),
        "branches": sum(1 for node in nodes if node["parent_id"] is not None),
        "completed": sum(1 for node in nodes if node["is_completed"]),
        "pruned_with_output": sum(1 for node in nodes if node["is_pruned"] and node["output_token_ids"] is not None),
        "pruned_without_output": sum(1 for node in nodes if node["is_pruned"] and node["output_token_ids"] is None),
        "active_at_end": sum(1 for node in nodes if node["is_active"]),
        "max_depth": max((node["branch_depth"] for node in nodes), default=0),
    }


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(temporary, path)


def audited(run_single_question: Callable) -> Callable:
    """Wrap benchmark_eval.run_single_question so each question also writes question_NNNN.audit.json."""

    def run(*, llm, tokenizer, question, ground_truth, sampling_params, question_index, output_dir, enable_branching=False):
        before = read_counters(llm)
        entry = run_single_question(
            llm=llm, tokenizer=tokenizer, question=question, ground_truth=ground_truth,
            sampling_params=sampling_params, question_index=question_index,
            output_dir=output_dir, enable_branching=enable_branching,
        )
        after = read_counters(llm)
        engine_tokens = generation_tokens(after) - generation_tokens(before)
        audit = {
            "question_index": question_index,
            "reported_total_tokens": entry["total_tokens"],
            "engine_generation_tokens": engine_tokens,
            "unaccounted_tokens": engine_tokens - entry["total_tokens"],
            "counter_deltas": {key: after[key] - before.get(key, 0.0) for key in after},
            "tree": summarize_tree(llm.get_gambit_tree_export() if enable_branching else None),
        }
        write_json_atomic(Path(output_dir) / f"question_{question_index:04d}.audit.json", audit)
        entry["audit"] = audit
        return entry

    return run


def main() -> None:
    args = parse_args(sys.argv[1:])
    forwarded = benchmark_eval_argv(args)
    print("benchmark_eval.py " + " ".join(forwarded), flush=True)
    benchmark_eval.run_single_question = audited(benchmark_eval.run_single_question)
    sys.argv = [benchmark_eval.__file__, *forwarded]
    benchmark_eval.main()


if __name__ == "__main__":
    main()
