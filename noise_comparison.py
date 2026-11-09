"""Report paired changes from adding Gumbel ordering noise."""

import csv
import json
from pathlib import Path

import numpy as np

from compare_models import MATCH_SETTINGS, load_outcomes, paired_interval
from experiment_settings import differing_settings
from sampling import METHODS, digit_description, noise_description

LABELS = {"random": "Random", "probability": "Probability",
          "margin": "Margin", "entropy": "Entropy"}


def compare_noise(folders, output):
    loaded = [load_outcomes(Path(folder)) for folder in folders]
    by_scale = {}
    for item in loaded:
        scale = item[0]["settings"]["gumbel_scale"]
        if not np.isfinite(scale) or scale < 0 or scale in by_scale:
            raise ValueError("Noise scales must be distinct, finite and nonnegative.")
        by_scale[scale] = item
    if 0 not in by_scale or len(by_scale) < 2:
        raise ValueError("Include zero noise and at least one nonzero noise scale.")
    baseline, reference, base_outcomes = by_scale[0]
    matching_keys = tuple(key for key in MATCH_SETTINGS if key != "gumbel_scale")
    matching_keys += ("device", "torch_version")
    for summary, arrays, _ in loaded:
        differences = differing_settings(summary["settings"], baseline["settings"], matching_keys)
        if differences:
            raise ValueError(f"Mismatched benchmark setting: {differences[0]}")
        if summary["model"] != baseline["model"]:
            raise ValueError("Noise comparisons must use exactly the same model checkpoint.")
        if summary["dataset"]["sha256"] != baseline["dataset"]["sha256"]:
            raise ValueError("Dataset checksums differ.")
        if not all(np.array_equal(a, b) for a, b in zip(reference, arrays)):
            raise ValueError("Noise comparisons require identical ordered puzzles and solutions.")

    rows = []
    for scale, (summary, _, outcomes) in sorted(by_scale.items()):
        for method in METHODS:
            per_puzzle = outcomes[method].mean(axis=0)
            accuracy, low, high = paired_interval(per_puzzle)
            effect, effect_low, effect_high = paired_interval(
                per_puzzle - base_outcomes[method].mean(axis=0))
            runs = [run for run in summary["runs"] if run["method"] == method]
            rows.append(dict(
                method=method, gumbel_scale=scale, exact_accuracy=accuracy,
                accuracy_ci_low=low, accuracy_ci_high=high,
                seed_accuracies=outcomes[method].mean(axis=1).tolist(),
                change_from_zero=effect, change_ci_low=effect_low, change_ci_high=effect_high,
                valid_rate=float(np.mean([r["valid_rate"] for r in runs])),
                blank_cell_accuracy=float(np.mean([r["blank_cell_accuracy"] for r in runs])),
                avg_passes=float(np.mean([r["avg_passes"] for r in runs])),
                decode_seconds=float(np.sum([r["decode_seconds"] for r in runs])),
            ))
        # Noise is ignored by the random rule: this is a useful experiment control.
        if not np.array_equal(outcomes["random"], base_outcomes["random"]):
            raise ValueError("Random-order outcomes changed; the noise control failed.")

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(model=baseline["model"], dataset=baseline["dataset"],
                    settings=baseline["settings"], scales=sorted(by_scale),
                    source_folders=[str(folder) for folder in folders],
                    bootstrap_repeats=10000, bootstrap_seed=20261105,
                    noise_convention=noise_description(baseline["settings"].get("noise_mode", "legacy-raw")),
                    rows=rows)
    (output / "noise_comparison.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with (output / "noise_comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Effect of ordering noise", "",
             f"Model: {baseline['model']['name']} at `{baseline['model']['revision']}`.",
             f"{len(reference[0]):,} identical puzzles; seeds {baseline['settings']['seeds']}; "
             f"{baseline['settings']['steps']} steps; {baseline['settings']['schedule']} reveal schedule.", "",
             "Noise changes cell ordering only. The digit-selection rule, clues, reveal budgets and seeds stay fixed.",
             f"Digit prediction: {digit_description(baseline['settings'].get('digit_mode', 'greedy'))}.",
             f"Noise: {metadata['noise_convention']}. Random ordering ignores this setting.", "",
             "| Method | Noise scale | Exact accuracy | Change from zero (pp) | 95% paired interval (pp) |",
             "| --- | --- | --- | --- | --- |"]
    for row in rows:
        lines.append(f"| {LABELS[row['method']]} | {row['gumbel_scale']:g} | {row['exact_accuracy']:.2%} | "
                     f"{100 * row['change_from_zero']:+.2f} | "
                     f"[{100 * row['change_ci_low']:+.2f}, {100 * row['change_ci_high']:+.2f}] |")
    lines += ["", "![Noise benchmark](noise_comparison.png)", "",
              "Intervals are descriptive 95% paired puzzle-bootstrap intervals (10,000 resamples). "
              "All seeds for a puzzle stay together; seeds do not increase the number of independent puzzles. "
              "They quantify puzzle sampling uncertainty conditional on this checkpoint and these inference seeds.", "",
              "The same noise scale affects probability, margin and negative entropy differently because "
              "their score domains differ. This measures sensitivity to our noise convention. "
              "It does not establish equivalence to the paper's exact sampler or explain the entire paper gap.", "",
              "The graph includes individual seed rates as dots. Paired changes are relative to each method's "
              "own zero-noise outcome. Timing is descriptive; a reused baseline may have run under different CPU load.", ""]
    (output / "noise_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    plot_noise(metadata, output)
    return metadata


def plot_noise(report, output):
    import os
    import tempfile

    # Font-cache writes must also work in restricted/offline project sessions.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sudoku-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scales = report["scales"]
    fig, (accuracy_ax, change_ax) = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    colors = plt.get_cmap("viridis")(np.linspace(.2, .8, len(scales)))
    positions = np.arange(len(METHODS))
    width = .8 / len(scales)
    for i, (scale, color) in enumerate(zip(scales, colors)):
        rows = [next(r for r in report["rows"] if r["method"] == method
                     and r["gumbel_scale"] == scale) for method in METHODS]
        xs = positions + (i - (len(scales) - 1) / 2) * width
        rates = np.array([row["exact_accuracy"] * 100 for row in rows])
        errors = np.array([[row["exact_accuracy"] - row["accuracy_ci_low"] for row in rows],
                           [row["accuracy_ci_high"] - row["exact_accuracy"] for row in rows]]) * 100
        bars = accuracy_ax.bar(xs, rates, width, color=color, label=f"Noise {scale:g}",
                               yerr=np.maximum(errors, 0), capsize=3)
        accuracy_ax.bar_label(bars, labels=[f"{rate:.1f}%" for rate in rates], padding=5, fontsize=8)
        for x, row in zip(xs, rows):
            seed_values = np.array(row["seed_accuracies"]) * 100
            offsets = np.linspace(-width * .15, width * .15, len(seed_values))
            accuracy_ax.scatter(x + offsets, seed_values, s=13, color="black", zorder=4, alpha=.65)
        if scale != 0:
            changes = np.array([row["change_from_zero"] * 100 for row in rows])
            errors = np.array([[row["change_from_zero"] - row["change_ci_low"] for row in rows],
                               [row["change_ci_high"] - row["change_from_zero"] for row in rows]]) * 100
            change_ax.errorbar(positions + (i - len(scales) / 2) * .12, changes,
                               yerr=np.maximum(errors, 0), fmt="o", capsize=4,
                               color=color, label=f"Noise {scale:g} minus 0")
    for axis in (accuracy_ax, change_ax):
        axis.set_xticks(positions, [LABELS[m] for m in METHODS])
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(fontsize=9)
    accuracy_ax.set(title="Solving accuracy", ylabel="Exact puzzle accuracy (%)", ylim=(0, 110))
    change_ax.axhline(0, color="black", linewidth=.8)
    change_ax.set(title="Paired impact of noise", ylabel="Accuracy change (percentage points)")
    n = len(report["dataset"]["selected_puzzle_ids"])
    fig.suptitle(f"{report['model']['name']} | {n:,} puzzles | "
                 f"{len(report['settings']['seeds'])} inference seeds", fontsize=14)
    fig.supxlabel("Gumbel ordering noise; error bars: 95% puzzle-bootstrap intervals; dots on bars: inference seeds", fontsize=9)
    for extension in ("png", "pdf"):
        fig.savefig(Path(output) / f"noise_comparison.{extension}", dpi=180)
    plt.close(fig)
