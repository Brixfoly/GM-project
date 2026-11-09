"""Benchmark and graph the same model with and without ordering noise."""

import argparse
import math
from pathlib import Path
import subprocess
import sys

from noise_comparison import compare_noise
from sampling import DIGIT_MODES, NOISE_MODES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="tchauffi/sudoku-dit")
    parser.add_argument("--revision", default="3b5e8349fc4fd122d8297da714f8004080d1361c")
    parser.add_argument("--noise-scales", type=float, nargs="+", default=[0, .5])
    parser.add_argument("--noise-mode", choices=NOISE_MODES, default="upstream")
    parser.add_argument("--digit-mode", choices=DIGIT_MODES, default="sample")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--output", type=Path, default=Path("results/noise_benchmark"))
    parser.add_argument("--baseline", type=Path, help="Reuse a matching zero-noise benchmark folder")
    args = parser.parse_args()
    if (len(set(args.noise_scales)) != len(args.noise_scales) or 0 not in args.noise_scales
            or len(args.noise_scales) < 2
            or any(not math.isfinite(s) or s < 0 for s in args.noise_scales)):
        parser.error("noise-scales must be distinct, finite, nonnegative and include 0 and a positive scale")
    if args.limit < 0 or min(args.steps, args.batch_size, args.threads) < 1:
        parser.error("limit must be nonnegative; steps, batch-size and threads must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct")
    args.output.mkdir(parents=True, exist_ok=True)
    folders = []
    for scale in sorted(args.noise_scales):
        tag = str(scale).removesuffix(".0").replace(".", "p")
        folder = args.baseline if scale == 0 and args.baseline else args.output / f"noise_{tag}"
        command = [sys.executable, str(Path(__file__).with_name("benchmark.py")),
                   "--model", args.model, "--revision", args.revision,
                   "--gumbel-scale", str(scale), "--noise-mode", args.noise_mode, "--limit", str(args.limit),
                   "--sample-seed", str(args.sample_seed), "--steps", str(args.steps),
                   "--batch-size", str(args.batch_size), "--threads", str(args.threads),
                   "--device", args.device, "--digit-mode", args.digit_mode,
                   "--seeds", *map(str, args.seeds),
                   "--output", str(folder), "--resume"]
        print(f"\nBenchmark: noise {scale:g}, artifacts {folder}", flush=True)
        subprocess.run(command, check=True)
        folders.append(folder)
    compare_noise(folders, args.output)
    print(f"\nGraph and paired benchmark report: {args.output / 'noise_comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
