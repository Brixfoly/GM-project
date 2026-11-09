"""Compare completed benchmarks, pairing puzzles and averaging inference seeds.

Example: python compare_models.py results/cross_model_20261105/*
Pass explicit model directories in PowerShell (it does not expand this glob).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from sampling import METHODS
from experiment_settings import differing_settings

CONTRASTS = (("probability", "random"), ("margin", "probability"),
             ("entropy", "probability"))
MATCH_SETTINGS = ("sample_seed", "seeds", "steps", "batch_size", "gumbel_scale",
                  "schedule", "protocol", "smoke_test", "noise_mode", "digit_mode")


def load_outcomes(folder):
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if summary["settings"]["smoke_test"]:
        raise ValueError("Smoke tests are not research comparisons.")
    if summary["settings"]["protocol"] != "paper-inspired":
        raise ValueError("Compare one scheduled budget per model with this report.")
    outcomes = {}
    reference = None
    for method in METHODS:
        per_seed = []
        for seed in summary["settings"]["seeds"]:
            runs = [r for r in summary["runs"] if r["method"] == method and r["seed"] == seed]
            if len(runs) != 1:
                raise ValueError(f"Missing or duplicate {method}/seed {seed} in {folder}")
            with np.load(folder / runs[0]["predictions_file"]) as data:
                arrays = tuple(data[key].copy() for key in ("puzzle_ids", "puzzles", "solutions"))
                if reference is None:
                    reference = arrays
                elif not all(np.array_equal(a, b) for a, b in zip(reference, arrays)):
                    raise ValueError("Puzzle IDs, inputs or solutions differ within a model.")
                per_seed.append((data["predictions"] == data["solutions"]).all(axis=1))
        outcomes[method] = np.asarray(per_seed, dtype=float)
    return summary, reference, outcomes


def paired_interval(differences, repeats=10000):
    """Bootstrap puzzle clusters; all seed outcomes for a puzzle stay together."""
    differences = np.asarray(differences, dtype=float)
    generator = np.random.default_rng(20261105)
    means = np.empty(repeats)
    for start in range(0, repeats, 100):
        size = min(100, repeats - start)
        indices = generator.integers(len(differences), size=(size, len(differences)))
        means[start:start + size] = differences[indices].mean(axis=1)
    return float(differences.mean()), *map(float, np.quantile(means, [.025, .975]))


def compare(folders, output, baseline="tchauffi/sudoku-dit"):
    loaded = [load_outcomes(folder) for folder in folders]
    first_summary, first_reference, _ = loaded[0]
    names = [summary["model"]["name"] for summary, _, _ in loaded]
    if len(set(names)) != len(names) or baseline not in names:
        raise ValueError("Use distinct models and include the requested baseline.")
    for summary, reference, _ in loaded[1:]:
        differences = differing_settings(summary["settings"], first_summary["settings"], MATCH_SETTINGS)
        if differences:
            raise ValueError(f"Mismatched benchmark setting: {differences[0]}")
        if summary["dataset"]["sha256"] != first_summary["dataset"]["sha256"]:
            raise ValueError("Dataset checksums differ.")
        if not all(np.array_equal(a, b) for a, b in zip(first_reference, reference)):
            raise ValueError("Models must use identical ordered puzzles and references.")

    baseline_outcomes = loaded[names.index(baseline)][2]
    n = len(first_reference[0])
    seeds = first_summary["settings"]["seeds"]
    lines = ["# Cross-model ordering experiment", "",
             f"{n:,} identical puzzles, inference seeds {seeds}.",
             f"Steps: {first_summary['settings']['steps']}; schedule: {first_summary['settings']['schedule']}; "
             f"ordering noise: {first_summary['settings']['gumbel_scale']}.", "",
             "Exact puzzle accuracy, averaged over seeds. Repeats reuse the same puzzles.", "",
             "| Model | Random | Probability | Margin | Entropy |",
             "| --- | --- | --- | --- | --- |"]
    numeric = []
    for summary, _, outcomes in loaded:
        name = summary["model"]["name"]
        rates = {method: float(outcomes[method].mean()) for method in METHODS}
        lines.append(f"| {name} | " + " | ".join(f"{rates[m]:.2%}" for m in METHODS) + " |")
        numeric.append({"model": summary["model"], "accuracy": rates, "contrasts": []})

    lines += ["", "## Paired ordering effects", "",
              "Differences in percentage points, with descriptive 95% paired puzzle-bootstrap intervals "
              "(10,000 resamples, fixed bootstrap seed). Seed outcomes are averaged within puzzle first. "
              "These intervals describe puzzle sampling uncertainty conditional on these checkpoints and "
              "inference seeds; they are not adjusted for multiple comparisons.", "",
              "| Model | Contrast | Effect (pp) | 95% interval | Change from original model (pp) | 95% interval |",
              "| --- | --- | --- | --- | --- | --- |"]
    for (summary, _, outcomes), record in zip(loaded, numeric):
        name = summary["model"]["name"]
        for a, b in CONTRASTS:
            delta = (outcomes[a] - outcomes[b]).mean(axis=0)
            base_delta = (baseline_outcomes[a] - baseline_outcomes[b]).mean(axis=0)
            effect, low, high = paired_interval(delta)
            interaction, ilo, ihi = paired_interval(delta - base_delta)
            lines.append(f"| {name} | {a} - {b} | {effect * 100:+.2f} | "
                         f"[{low * 100:+.2f}, {high * 100:+.2f}] | {interaction * 100:+.2f} | "
                         f"[{ilo * 100:+.2f}, {ihi * 100:+.2f}] |")
            record["contrasts"].append({"a": a, "b": b, "effect": effect, "ci": [low, high],
                                        "change_from_baseline": interaction, "change_ci": [ilo, ihi]})
    training_comparisons = []
    if "harry-500k" in names:
        encoder_baseline = loaded[names.index("harry-500k")][2]
        lines += ["", "## Same-architecture checkpoint comparisons", "",
                  "Relative to harry-500k. Accuracy differences and changes in ordering effects "
                  "are paired over the same puzzles and seeds. This removes the architecture "
                  "difference, while initialization, optimization and exact data histories remain uncontrolled.", "",
                  "| Checkpoint | Quantity | Change (pp) | 95% paired interval |",
                  "| --- | --- | --- | --- |"]
        for summary, _, outcomes in loaded:
            name = summary["model"]["name"]
            if not name.startswith("harry-") or name == "harry-500k":
                continue
            quantities = [("probability accuracy", outcomes["probability"].mean(axis=0)
                           - encoder_baseline["probability"].mean(axis=0))]
            for a, b in CONTRASTS[1:]:
                difference = ((outcomes[a] - outcomes[b])
                              - (encoder_baseline[a] - encoder_baseline[b])).mean(axis=0)
                quantities.append((f"{a} - {b} effect", difference))
            for label, difference in quantities:
                effect, low, high = paired_interval(difference)
                lines.append(f"| {name} | {label} | {effect * 100:+.2f} | "
                             f"[{low * 100:+.2f}, {high * 100:+.2f}] |")
                training_comparisons.append(dict(model=name, baseline="harry-500k", quantity=label,
                                                 change=effect, ci=[low, high]))
    lines += ["", "## What this experiment can establish", "",
              "- Changing checkpoints tests whether ordering effects generalize across the evaluated models.",
              "- The linked checkpoints share an architecture. Differences between them are consistent with "
              "training effects, but training data, initialization and optimization were not experimentally controlled.",
              "- Comparing the original DiT against the linked encoder changes both architecture and training; "
              "it cannot isolate a causal effect of training alone.",
              "- Training/test overlap is unaudited. The checkpoint filenames do not fully establish their training histories.",
              "- The common decoder maps the encoder's mask token and normalizes only digits 1..9. "
              "It ignores timestep for the encoder and preserves all revealed predictions. This is not upstream sampler reproduction.",
              "- A near-zero ordering contrast does not establish equivalence; low overall performance can hide differences.",
              "- Benchmarks may run concurrently. Their CPU timings are not suitable for ranking model speed.",
              "", "## Provenance", ""]
    for folder, (summary, _, _) in zip(folders, loaded):
        info = summary["model"]
        lines.append(f"- {info['name']}: revision `{info['revision']}`, "
                     f"{info['parameters']:,} parameters; artifacts: `{folder}`.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "comparison.json").write_text(json.dumps({
        "baseline": baseline, "puzzles": n, "seeds": seeds,
        "bootstrap_repeats": 10000, "bootstrap_seed": 20261105, "models": numeric,
        "same_architecture_comparisons": training_comparisons,
    }, indent=2), encoding="utf-8")
    print(f"Cross-model report saved in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("results/cross_model_comparison"))
    parser.add_argument("--baseline", default="tchauffi/sudoku-dit")
    args = parser.parse_args()
    compare(args.folders, args.output, args.baseline)
