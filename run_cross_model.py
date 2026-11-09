"""Run the five-model experiment locally, saving separate logs and resuming progress."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys

from compare_models import compare
from sampling import DIGIT_MODES

MODELS = ("harry-100k", "harry-500k", "harry-combined", "harry-hard", "tchauffi/sudoku-dit")
DIT_REVISION = "3b5e8349fc4fd122d8297da714f8004080d1361c"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/cross_model_sample"))
    parser.add_argument("--digit-mode", choices=DIGIT_MODES, default="sample")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or args.threads < 1:
        parser.error("workers and threads must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    def run(model):
        folder = args.output / model.replace("/", "_")
        command = [sys.executable, "benchmark.py", "--model", model, "--revision", DIT_REVISION,
                   "--limit", "1000", "--seeds", "0", "1", "2", "--steps", "50",
                   "--batch-size", "32", "--gumbel-scale", "0", "--threads", str(args.threads),
                   "--digit-mode", args.digit_mode, "--output", str(folder), "--resume"]
        print(f"Running {model}; log: {folder.with_suffix('.log')}", flush=True)
        with folder.with_suffix(".log").open("a", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"Completed {model}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, model) for model in MODELS]
        for future in as_completed(futures):
            future.result()
    compare([args.output / model.replace("/", "_") for model in MODELS], args.output)


if __name__ == "__main__":
    main()
