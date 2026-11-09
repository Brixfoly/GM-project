"""Audit saved benchmarks against the paper; never infer missing author details.

This checks compatibility, not agreement with the published accuracy numbers.
An exact-reproduction claim requires provenance that the public specification
and the currently supported models do not supply.
"""

import argparse
import json
from pathlib import Path

SPEC_PATH = Path(__file__).with_name("paper_requirements.json")


def audit_summary(summary, spec=None):
    if spec is None:
        spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    target = spec["requirements_stated_in_paper"]
    settings = summary["settings"]
    model = summary["model"]
    checks = []

    def check(item, expected, observed, status, detail=""):
        checks.append(dict(item=item, expected=expected, observed=observed,
                           status=status, detail=detail))

    scheduled = settings.get("protocol") == "paper-inspired"
    steps_match = scheduled and settings.get("steps") == target["reverse_sampling_steps"]
    check("Reverse sampling budget", target["reverse_sampling_steps"],
          settings.get("steps") if scheduled else settings.get("protocol"),
          "matched" if steps_match else "mismatch", "This matches the step count only.")
    digit_mode = settings.get("digit_mode", "greedy")
    check("Digit value selection", "sample", digit_mode,
          "matched" if digit_mode == "sample" else "mismatch",
          "The paper samples each selected token from p_theta; historical runs used argmax.")
    noise = settings.get("gumbel_scale")
    check("Gumbel coefficient", target["gumbel_coefficient"], noise,
          "matched" if noise == target["gumbel_coefficient"] else "mismatch",
          "A matching coefficient does not verify the score domain, annealing or remasking.")
    checksum = summary["dataset"].get("sha256")
    check("Cited standard test-data release", spec["verified_test_release_sha256"], checksum,
          "matched" if checksum == spec["verified_test_release_sha256"] else "unverified",
          "Release identity does not establish the authors' evaluated row IDs.")
    parameters = model.get("parameters")
    compatible_size = isinstance(parameters, (int, float)) and 5_000_000 <= parameters <= 7_000_000
    check("Approximate model scale", "roughly 6M parameters", parameters,
          "unverified" if compatible_size or parameters is None else "mismatch",
          "A compatible parameter count alone cannot identify the architecture or training.")
    name = model.get("name", "")
    known_independent = name.startswith("harry-") or name == "tchauffi/sudoku-dit"
    check("Paper's trained model", target["model_family"], name,
          "mismatch" if known_independent else "unverified",
          "The DiT and linked encoder checkpoints are different architectures and training runs.")
    observed_methods = sorted({run["method"] for run in summary.get("runs", [])})
    required = target["ordering_methods"]
    check("Required ordering strategies", required, observed_methods,
          "matched" if set(required) <= set(observed_methods) else "mismatch",
          "Matching names does not verify sampler behavior. Entropy is an extension.")
    check("Training provenance", dict(learning_rate=target["learning_rate"],
          epochs=target["training_epochs"], batch_size_as_reported=target["batch_size_as_reported"],
          dataset=target["training_dataset"]), summary.get("training_provenance"), "unverified")
    check("Authors' complete inference implementation", "verified author sampler and tokenizer",
          summary.get("implementation"), "unverified")
    ids = summary["dataset"].get("selected_puzzle_ids", [])
    check("Authors' evaluated instances", "author-confirmed row IDs", f"{len(ids)} local rows",
          "unverified", "The paper does not identify its exact evaluated rows or seeds.")
    return dict(
        paper=spec["version"], target=spec["target"], model=model,
        exact_reproduction_verified=False,
        status="not an exact reproduction" if any(c["status"] == "mismatch" for c in checks)
               else "exact reproduction unverified",
        checks=checks, unresolved=spec["unresolved_for_exact_reproduction"],
        interpretation="Do not tune settings to the published percentages or treat matching accuracy as provenance.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", type=Path, nargs="+", help="Benchmark summary.json paths")
    parser.add_argument("--output", type=Path, default=Path("results/paper_audit"))
    parser.add_argument("--require-exact", action="store_true",
                        help="Exit unsuccessfully unless exact reproduction is verified")
    args = parser.parse_args()
    audits = []
    for path in args.summaries:
        summary = json.loads(path.read_text(encoding="utf-8"))
        result = audit_summary(summary)
        result["summary_file"] = str(path)
        audits.append(result)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
    lines = ["# Paper reproduction audit", "",
             "Target: arXiv:2502.06768v3, Sudoku Table 2 and Appendix D.2 (PDF page 19).", "",
             "A matched scalar or method name does not verify the full experiment.", ""]
    for audit in audits:
        lines += [f"## {audit['summary_file']}", "", f"Status: **{audit['status']}**.", "",
                  "| Check | Status | Observed | Detail |", "| --- | --- | --- | --- |"]
        for item in audit["checks"]:
            observed = json.dumps(item["observed"], ensure_ascii=False).replace("|", "\\|")
            lines.append(f"| {item['item']} | {item['status']} | {observed} | {item['detail']} |")
        lines += ["", "Unresolved requirements:", ""]
        lines += [f"- {item}" for item in audit["unresolved"]]
        lines += [""]
        print(f"{audit['summary_file']}: {audit['status']}")
    (args.output / "audit.md").write_text("\n".join(lines), encoding="utf-8")
    if args.require_exact and not all(a["exact_reproduction_verified"] for a in audits):
        parser.exit(2, "Exact reproduction is not verified. See the audit and PAPER_REPRODUCTION.md.\n")


if __name__ == "__main__":
    main()
