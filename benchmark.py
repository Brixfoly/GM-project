"""Run the Sudoku ordering experiment on the original NumPy test data."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiment_settings import differing_settings
from prepare_data import DEFAULT_DATA, load_example, load_test_data
from reporting import evaluate, save_report
from sampling import DIGIT_MODES, METHODS, NOISE_MODES, solve

RESUME_SETTINGS = (
    "model", "revision", "limit", "sample_seed", "seeds", "steps",
    "batch_size", "gumbel_scale", "schedule", "protocol", "k",
    "smoke_test", "device", "torch_version", "noise_mode", "digit_mode",
)


def choose_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def wait_for_device(device):
    """GPU operations must finish before we read the elapsed time."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def load_model(model_name, revision, device):
    from other_models import CHECKPOINTS, load_other_model

    if model_name in CHECKPOINTS:
        return load_other_model(model_name, device)

    from huggingface_hub import snapshot_download
    from nonet.model import SudokuDiT
    from safetensors.torch import load_file

    folder = Path(snapshot_download(
        model_name,
        revision=revision,
        allow_patterns=["config.json", "model.safetensors"],
    ))
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))

    # The published config includes a scheduler setting, not a model parameter.
    config.pop("schedule", None)
    model = SudokuDiT(**config)
    weights = load_file(str(folder / "model.safetensors"))
    model.load_state_dict(weights, strict=True)
    model = model.to(device).eval()

    information = {
        "name": model_name,
        "revision": folder.name,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    return model, information


def run_method(model, puzzles, method, seed, cells_per_step, args, device):
    """Solve batches, measuring inference time but not data transfers."""
    all_predictions = []
    all_passes = []
    seconds = 0.0

    for start in range(0, len(puzzles), args.batch_size):
        batch = torch.from_numpy(puzzles[start:start + args.batch_size]).to(device)

        wait_for_device(device)
        before = time.perf_counter()
        predictions, passes = solve(
            model, batch, method,
            seed=seed + start,
            steps=args.steps,
            cells_per_step=cells_per_step,
            schedule=args.schedule,
            gumbel_scale=args.gumbel_scale,
            noise_mode=args.noise_mode,
            digit_mode=args.digit_mode,
        )
        wait_for_device(device)
        seconds += time.perf_counter() - before

        all_predictions.append(predictions.cpu().numpy())
        all_passes.append(passes.cpu().numpy())

        if (start // args.batch_size + 1) % 50 == 0:
            completed = min(start + args.batch_size, len(puzzles))
            print(f"  {method}, seed {seed}: {completed}/{len(puzzles)} puzzles", flush=True)

    return np.concatenate(all_predictions), np.concatenate(all_passes), seconds


def read_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--limit", type=int, default=1000, help="Number of puzzles; 0 uses all")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--seeds", "--seed", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=None, help="CPU torch threads")
    parser.add_argument("--gumbel-scale", type=float, default=0.5,
                        help="Gumbel coefficient (paper: 0.5; 0 disables noise)")
    parser.add_argument("--noise-mode", choices=NOISE_MODES, default="upstream")
    parser.add_argument("--digit-mode", choices=DIGIT_MODES, default="sample",
                        help="Sample digits from p_theta (paper), or use historical greedy argmax")
    parser.add_argument("--schedule", choices=["binomial", "linear"], default="binomial")
    parser.add_argument("--protocol", choices=["paper-inspired", "fixed-k"], default="paper-inspired")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--model", default="tchauffi/sudoku-dit",
                        help="Hugging Face DiT or harry-100k/500k/combined/hard")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=Path, default=Path("results/benchmark"), help="Results folder")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse matching completed runs")
    args = parser.parse_args()

    if args.limit < 0 or args.steps < 1 or args.batch_size < 1:
        parser.error("limit must be nonnegative; steps and batch-size must be positive")
    if args.threads is not None and args.threads < 1:
        parser.error("threads must be positive")
    if any(count < 1 or count > 81 for count in args.k):
        parser.error("k must be between 1 and 81")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Inference seeds must be distinct")
    if not 0 <= args.gumbel_scale < float("inf"):
        parser.error("gumbel-scale must be finite and nonnegative")

    return args


def load_previous_results(output, settings, model_info, data_info):
    """Load progress only when it describes the same experiment."""
    summary_path = output / "summary.json"
    if not settings["resume"] or not summary_path.exists():
        return []

    previous = json.loads(summary_path.read_text(encoding="utf-8"))
    # Output path, thread count and resume mode do not change the experiment.
    differences = differing_settings(previous["settings"], settings, RESUME_SETTINGS)
    if differences or previous["model"] != model_info or previous["dataset"] != data_info:
        raise ValueError("Cannot resume: settings, model or dataset differ.")
    return previous["runs"]


def reuse_completed_run(results, output, method, seed, cells_per_step,
                       puzzles, solutions, puzzle_ids):
    """Validate an existing run before allowing the benchmark to skip it."""
    completed = [run for run in results if run["method"] == method
                 and run["seed"] == seed and run["k"] == cells_per_step]
    if not completed:
        return False
    if len(completed) != 1:
        raise ValueError("Duplicate completed run in summary.")

    with np.load(output / completed[0]["predictions_file"]) as saved:
        expected_arrays = {"puzzle_ids": puzzle_ids, "puzzles": puzzles, "solutions": solutions}
        for key, expected in expected_arrays.items():
            if not np.array_equal(saved[key], expected):
                raise ValueError(f"Cannot resume: saved {key} differ.")
        if (saved["predictions"].shape != puzzles.shape
                or saved["passes"].shape != (len(puzzles),)):
            raise ValueError("Cannot resume: invalid prediction artifact shapes.")
    return True


def main():
    args = read_arguments()
    if args.threads is not None:
        torch.set_num_threads(args.threads)

    # 1. Read the original .npy file and reconstruct the selected boards in memory.
    if args.smoke_test:
        puzzles, solutions, puzzle_ids, data_info = load_example()
        print("SMOKE TEST: one example, not a research benchmark.")
    else:
        puzzles, solutions, puzzle_ids, data_info = load_test_data(
            args.data, args.limit, args.sample_seed,
        )

    # 2. Load one frozen model, reused by every method.
    device = choose_device(args.device)
    torch.manual_seed(args.seeds[0])
    print(f"Loading {args.model} on {device}...", flush=True)
    model, model_info = load_model(args.model, args.revision, device)

    with torch.inference_mode():
        model(torch.from_numpy(puzzles[:1]).to(device), torch.ones(1, device=device))
    wait_for_device(device)

    # 3. Run each method on the same puzzles, budgets, and inference seeds.
    args.output.mkdir(parents=True, exist_ok=True)
    settings = vars(args).copy()
    settings["device"] = str(device)
    settings["torch_version"] = torch.__version__
    settings["threads"] = torch.get_num_threads()
    results = load_previous_results(args.output, settings, model_info, data_info)
    budgets = args.k if args.protocol == "fixed-k" else [None]

    for cells_per_step in budgets:
        for seed in args.seeds:
            for method in METHODS:
                if reuse_completed_run(
                    results, args.output, method, seed, cells_per_step,
                    puzzles, solutions, puzzle_ids,
                ):
                    print(f"Reusing {method}, seed {seed}", flush=True)
                    continue
                predictions, passes, seconds = run_method(
                    model, puzzles, method, seed, cells_per_step, args, device,
                )
                metrics = evaluate(puzzles, solutions, predictions, passes, seconds)

                budget_name = f"k{cells_per_step}" if cells_per_step else f"steps{args.steps}"
                filename = f"{method}_seed{seed}_{budget_name}.npz"
                np.savez_compressed(
                    args.output / filename,
                    puzzle_ids=puzzle_ids,
                    puzzles=puzzles.astype(np.int8),
                    solutions=solutions.astype(np.int8),
                    predictions=predictions.astype(np.int8),
                    passes=passes,
                )

                result = {
                    "method": method,
                    "seed": seed,
                    "k": cells_per_step,
                    "steps": args.steps if cells_per_step is None else None,
                    "predictions_file": filename,
                    **metrics,
                }
                results.append(result)
                print(
                    f"{method:12} seed={seed} {budget_name:8} "
                    f"solved={metrics['exact_match_rate']:.2%} "
                    f"valid={metrics['valid_rate']:.2%} "
                    f"time={seconds:.2f}s",
                    flush=True,
                )

                # 4. Save progress after each method, including the paper comparison.
                save_report(args.output, settings, data_info, model_info, results)

    print(f"Results saved in {args.output}")


if __name__ == "__main__":
    main()
