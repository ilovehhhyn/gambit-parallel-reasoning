"""Checks that one audited_eval.py run directory shows every stage of the method working, and prints its key numbers.

Usage: python repro/check_smoke.py RUN_DIR METHOD NUM_TRACES
Exits 1 and lists every failed requirement when any stage did not happen.
"""

import json
import sys
from pathlib import Path

METHODS = ("parallel-sampling", "step", "gambit")


def requirements(method: str, num_traces: int, question: dict, audit: dict) -> list[tuple[str, bool]]:
    """Return (requirement, holds) pairs for one question of one method."""
    traces = list(question["generation_results"].values())
    scored = [trace for trace in traces if trace["final_score"] is not None]
    common = [
        ("the engine generated tokens", audit["engine_generation_tokens"] > 0),
        ("the evaluator marked at least one trace correct", question["num_correct_traces"] >= 1),
        ("the vote produced a final answer", question["final_answer"] is not None),
    ]
    if method == "parallel-sampling":
        return common + [
            (f"all {num_traces} traces completed", question["total_traces"] == num_traces),
            ("the script's token count equals the engine counter", audit["unaccounted_tokens"] == 0),
        ]
    if method == "step":
        return common + [("every completed trace has a scorer score", len(scored) == len(traces) > 0)]
    if method == "gambit":
        # The token gap is printed, not required: the counter lives in the front-end process,
        # and Gambit's drain loop can return before the front end reads the engine's last outputs.
        tree = audit["tree"]
        return common + [
            ("every completed trace has a scorer score", len(scored) == len(traces) > 0),
            ("the tournament branched at least once", tree["branches"] >= 1),
            ("the tournament pruned at least once", tree["pruned_with_output"] + tree["pruned_without_output"] >= 1),
        ]
    raise SystemExit(f"unknown method `{method}`; expected one of {', '.join(METHODS)}")


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    run_dir, method, num_traces = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    summary = json.loads((run_dir / "benchmark_summary.json").read_text())
    failures: list[str] = []
    for entry in summary["questions"]:
        index = entry["question_index"]
        question = json.loads((run_dir / f"question_{index:04d}.json").read_text())
        audit = json.loads((run_dir / f"question_{index:04d}.audit.json").read_text())
        print(f"question {index}: answer={question['final_answer']} truth={question['ground_truth']} "
              f"correct_traces={question['num_correct_traces']}/{question['total_traces']} "
              f"seconds={question['generation_time']:.0f} reported_tokens={audit['reported_total_tokens']} "
              f"engine_tokens={audit['engine_generation_tokens']:.0f} unaccounted={audit['unaccounted_tokens']:.0f}")
        if audit["tree"] is not None:
            print(f"question {index}: tree={json.dumps(audit['tree'], sort_keys=True)}")
        for requirement, holds in requirements(method, num_traces, question, audit):
            print(f"  [{'ok' if holds else 'FAIL'}] {requirement}")
            if not holds:
                failures.append(f"question {index}: {requirement}")
    if summary["num_questions"] == 0:
        failures.append("the run contains no questions")
    if failures:
        raise SystemExit("smoke check failed:\n" + "\n".join(failures))
    print(f"SMOKE_OK {method}")


if __name__ == "__main__":
    main()
