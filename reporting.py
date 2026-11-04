"""Measure Sudoku solutions and write a short comparison with the paper."""

import json
import math
import statistics
from collections import defaultdict

import numpy as np

from prepare_data import valid_grid

PAPER_URL = "https://arxiv.org/html/2502.06768v3"
PAPER_ACCURACY = {"random": 0.0688, "probability": 0.1851, "margin": 0.8949}


def evaluate(puzzles, solutions, predictions, passes, seconds):
    """Score only after solving. References never enter the solving loop."""
    originally_blank = puzzles == 0
    correct_cells = predictions == solutions
    exact_matches = correct_cells.all(axis=1)
    clues_preserved = ((predictions == puzzles) | originally_blank).all(axis=1)

    valid_solutions = []
    for index, prediction in enumerate(predictions):
        valid_solutions.append(valid_grid(prediction) and bool(clues_preserved[index]))

    return {
        "puzzles": len(puzzles),
        "exact_match_rate": float(exact_matches.mean()),
        "valid_rate": float(np.mean(valid_solutions)),
        "blank_cell_accuracy": float(correct_cells[originally_blank].mean()),
        "avg_passes": float(passes.mean()),
        "decode_seconds": seconds,
        "seconds_per_puzzle": seconds / len(puzzles),
    }


def wilson_interval(rate, count):
    """95% interval for a solving rate measured over one set of puzzles."""
    z = 1.959963984540054
    denominator = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / denominator
    half_width = z * math.sqrt(
        rate * (1 - rate) / count + z * z / (4 * count * count)
    ) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def save_report(folder, settings, data_info, model_info, results):
    """Keep numeric results in JSON and a readable comparison in Markdown."""
    summary = {
        "settings": settings,
        "dataset": data_info,
        "model": model_info,
        "runs": results,
    }
    (folder / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8",
    )

    groups = defaultdict(list)
    for result in results:
        groups[(result["method"], result["k"], result["steps"])].append(result)

    lines = [
        "# Sudoku ordering comparison",
        "",
        "Descriptive comparison with Table 2, not an exact reproduction.",
        f"Paper: {PAPER_URL}",
        "",
        f"Model: {model_info['name']} at {model_info['revision']}, "
        f"{model_info['parameters']:,} parameters.",
        f"Data: {data_info['path']}; {results[0]['puzzles']:,} puzzles per seed.",
        f"Verified original test release: {data_info.get('verified_original_test_set', False)}.",
        f"Schedule: {settings['schedule']}; raw-score Gumbel scale: {settings['gumbel_scale']}.",
        "",
        "| Method | Budget | Our exact accuracy (mean +/- seed SD) | Valid rate | Paper |",
        "| --- | --- | --- | --- | --- |",
    ]

    for (method, k, steps), runs in groups.items():
        accuracies = [run["exact_match_rate"] for run in runs]
        mean_accuracy = statistics.mean(accuracies)
        seed_std = statistics.stdev(accuracies) if len(runs) > 1 else 0.0
        valid_rate = statistics.mean(run["valid_rate"] for run in runs)
        paper_rate = PAPER_ACCURACY.get(method)
        paper_text = f"{paper_rate:.2%}" if paper_rate is not None else "Not reported"
        budget = f"{steps} steps" if steps is not None else f"k={k}"
        lines.append(
            f"| {method} | {budget} | {mean_accuracy:.2%} +/- {100 * seed_std:.2f} pp "
            f"| {valid_rate:.2%} | {paper_text} |"
        )

    lines += [
        "", "## Per-seed uncertainty", "",
        "| Method | Seed | Budget | Exact accuracy | 95% Wilson interval |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in results:
        low, high = wilson_interval(result["exact_match_rate"], result["puzzles"])
        budget = f"{result['steps']} steps" if result["steps"] is not None else f"k={result['k']}"
        lines.append(
            f"| {result['method']} | {result['seed']} | {budget} "
            f"| {result['exact_match_rate']:.2%} | {low:.2%} to {high:.2%} |"
        )

    lines += [
        "", "## Interpretation", "",
        "- Repeated seeds reuse the same puzzles; do not pool them as independent test instances.",
        "- The paper uses a different, approximately 6M-parameter model and training distribution. Our model's training overlap is unknown.",
        "- This is the cited standard test-data release, not the separate hard set. Exact evaluated author row IDs are unconfirmed.",
        "- Our optional Gumbel noise is added to raw scores. The cited upstream code uses log-probability scores, annealing and remasking. The exact modified author sampler was not located.",
        "- Our decoder preserves revealed predictions and uses greedy digit values. Noise scales are not equivalent across the adaptive scores.",
        "- The binomial reveal schedule with uniform cell selection has the vanilla independent-unmasking distribution.",
        "- Entropy is our extension. It has no published baseline in this paper.",
        "- Timing excludes loading, warmup, data transfers, evaluation and saving; it measures batched throughput.",
        "",
    ]
    (folder / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
