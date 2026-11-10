"""Small offline checks for the data reader and solving experiment."""

import tempfile
import json
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch

from prepare_data import EXAMPLE_SOLUTION, decode_records, load_example, load_test_data, valid_grid
from reporting import evaluate, wilson_interval
from sampling import METHODS, add_ordering_noise, choose_cells, make_reveal_schedule, score_cells, solve
from other_models import DigitAdapter
from compare_models import compare, paired_interval
from noise_comparison import compare_noise
from benchmark import load_previous_results, reuse_completed_run


class ContextModel(torch.nn.Module):
    """A fake model whose predictions change when the board changes."""
    def forward(self, board, timestep):
        digits = (torch.arange(81)[None, :] + board.sum(dim=1, keepdim=True)) % 9
        return torch.nn.functional.one_hot(digits, 9).float() * 3


class BenchmarkTests(unittest.TestCase):
    def test_resume_checks_settings_and_accepts_historical_noise_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            settings = dict(resume=True, noise_mode="legacy-raw", steps=50)
            model = dict(name="test", revision="fixed")
            dataset = dict(sha256="same")
            self.assertEqual(load_previous_results(output, settings, model, dataset), [])

            # Historical summaries omitted noise_mode; they used legacy-raw.
            summary = dict(settings=dict(steps=50), model=model, dataset=dataset,
                           runs=[dict(method="margin", seed=0, k=None)])
            (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            self.assertEqual(load_previous_results(output, settings, model, dataset), summary["runs"])
            for changes in [dict(noise_mode="upstream"), dict(steps=10),
                            dict(digit_mode="sample")]:
                with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "Cannot resume"):
                    load_previous_results(output, {**settings, **changes}, model, dataset)
            with self.assertRaisesRegex(ValueError, "Cannot resume"):
                load_previous_results(output, settings, {**model, "revision": "different"}, dataset)
            with self.assertRaisesRegex(ValueError, "Cannot resume"):
                load_previous_results(output, settings, model, dict(sha256="different"))
            self.assertEqual(
                load_previous_results(output, {**settings, "resume": False}, model, dataset), [],
            )

    def test_resume_validates_completed_prediction_artifacts(self):
        puzzles, solutions, puzzle_ids, _ = load_example()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = dict(method="margin", seed=0, k=None, predictions_file="run.npz")
            arrays = dict(puzzles=puzzles, solutions=solutions, puzzle_ids=puzzle_ids,
                          predictions=solutions, passes=np.array([50]))

            def reuse(results):
                return reuse_completed_run(
                    results, output, "margin", 0, None, puzzles, solutions, puzzle_ids,
                )

            self.assertFalse(reuse([]))
            np.savez(output / "run.npz", **arrays)
            self.assertTrue(reuse([result]))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                reuse([result, result])
            invalid_arrays = [dict(puzzle_ids=puzzle_ids + 1), dict(puzzles=solutions),
                              dict(solutions=puzzles), dict(predictions=solutions[:, :80]),
                              dict(passes=np.array([]))]
            for changes in invalid_arrays:
                with self.subTest(keys=list(changes)), self.assertRaisesRegex(ValueError, "Cannot resume"):
                    np.savez(output / "run.npz", **{**arrays, **changes})
                    reuse([result])

    def test_noise_matches_upstream_formula_and_zero_consumes_no_rng(self):
        scores = torch.tensor([[-2., -1., -.1]])
        generator = torch.Generator().manual_seed(9)
        state = generator.get_state().clone()
        self.assertTrue(torch.equal(add_ordering_noise(scores, 0, generator), scores))
        self.assertTrue(torch.equal(generator.get_state(), state))
        uniform = torch.rand(scores.shape, generator=torch.Generator().manual_seed(9))
        expected = scores + .3 * -torch.log(-torch.log(uniform + 1e-8) + 1e-8)
        self.assertTrue(torch.equal(add_ordering_noise(scores, .3, generator), expected))

    def test_solver_uses_log_probability_and_next_time_noise_rate(self):
        puzzles, _, _, _ = load_example()
        board = torch.from_numpy(puzzles)
        observed = []

        def capture(scores, scale, generator):
            observed.append((scores.clone(), scale))
            return scores

        with patch("sampling.add_ordering_noise", side_effect=capture):
            solve(ContextModel(), board, "probability", steps=4, schedule="linear")
        self.assertEqual([scale for _, scale in observed], [.375, .25, .125, 0.])
        expected = ContextModel()(board, None).log_softmax(-1).max(-1).values
        torch.testing.assert_close(observed[0][0], expected)
        with patch("sampling.add_ordering_noise", side_effect=AssertionError):
            solve(ContextModel(), board, "random", steps=4, schedule="linear")

    def setUp(self):
        puzzles, solutions, _, _ = load_example()
        self.puzzles = torch.from_numpy(puzzles)
        self.solutions = solutions

    def test_uncertainty_rankings(self):
        probabilities = torch.tensor([[[.50, .49, .01], [.45, .28, .27]]])
        expected_cells = {"probability": 0, "margin": 1, "entropy": 0}
        for method, expected in expected_cells.items():
            scores = score_cells(probabilities, method, generator=None)
            self.assertEqual(scores.argmax().item(), expected)

    def test_cell_selection_handles_ties_masks_and_different_counts(self):
        scores = torch.tensor([[9., 2., 2., 1.], [4., 3., 2., 1.], [1., 2., 3., 4.]])
        masked = torch.tensor([[False, True, True, True], [True, False, True, False],
                               [True, True, True, True]])
        selected = choose_cells(scores, masked, torch.tensor([1, 5, 0]))
        expected = torch.tensor([[False, True, False, False], [True, False, True, False],
                                 [False, False, False, False]])
        self.assertTrue(torch.equal(selected, expected))

    def test_completed_boards_do_not_call_model(self):
        class UnusedModel(torch.nn.Module):
            def forward(self, board, timestep):
                raise AssertionError("A completed batch does not need predictions.")

        boards = torch.from_numpy(self.solutions)
        for cells_per_step in [None, 5]:
            predictions, passes = solve(UnusedModel(), boards, "margin",
                                        cells_per_step=cells_per_step)
            self.assertTrue(torch.equal(predictions, boards))
            self.assertEqual(passes.item(), 0)

    def test_invalid_solver_settings_fail_early(self):
        invalid_settings = [dict(steps=0), dict(cells_per_step=0), dict(cells_per_step=82),
                            dict(gumbel_scale=-1), dict(gumbel_scale=float("nan")),
                            dict(digit_mode="invalid")]
        for settings in invalid_settings:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                solve(ContextModel(), self.puzzles, "margin", **settings)

    def test_digits_are_sampled_from_model_distribution_not_argmax(self):
        class TwoDigitModel(torch.nn.Module):
            def forward(self, board, timestep):
                logits = torch.full((*board.shape, 9), -float("inf"), device=board.device)
                logits[..., 0] = np.log(.75)
                logits[..., 1] = np.log(.25)
                return logits

        puzzles = torch.zeros((256, 81), dtype=torch.long)
        # A clue outside the predicted distribution must still be preserved.
        puzzles[:, 0] = 9
        model = TwoDigitModel()
        sampled, _ = solve(model, puzzles, "probability", cells_per_step=81, seed=17)
        repeated, _ = solve(model, puzzles, "probability", cells_per_step=81, seed=17)
        different, _ = solve(model, puzzles, "probability", cells_per_step=81, seed=18)
        self.assertTrue(torch.equal(sampled, repeated))
        self.assertFalse(torch.equal(sampled, different))
        self.assertTrue((sampled[:, 0] == 9).all())
        digits = sampled[:, 1:]
        self.assertTrue(((digits == 1) | (digits == 2)).all())
        self.assertAlmostEqual((digits == 2).float().mean().item(), .25, delta=.015)
        greedy, _ = solve(model, puzzles, "probability", cells_per_step=81,
                          seed=17, digit_mode="greedy")
        self.assertTrue((greedy[:, 1:] == 1).all())

    def test_sampling_only_draws_for_selected_cells(self):
        original = torch.multinomial
        draws = []

        def capture(probabilities, *args, **kwargs):
            draws.append(probabilities.shape[0])
            return original(probabilities, *args, **kwargs)

        with patch("sampling.torch.multinomial", side_effect=capture):
            solve(ContextModel(), self.puzzles, "margin", cells_per_step=5)
        blanks = int((self.puzzles == 0).sum())
        self.assertEqual(sum(draws), blanks)
        self.assertTrue(all(0 < count <= 5 for count in draws))

    def test_fixed_k_preserves_clues_and_uses_equal_budgets(self):
        second_puzzle = self.puzzles.clone()
        second_puzzle[0, :9] = torch.tensor([int(digit) for digit in EXAMPLE_SOLUTION[:9]])
        puzzles = torch.cat([self.puzzles, second_puzzle])

        for method in METHODS:
            predictions, passes = solve(ContextModel(), puzzles, method, cells_per_step=5, seed=42)
            self.assertTrue(torch.equal(predictions[puzzles != 0], puzzles[puzzles != 0]))
            self.assertTrue((predictions > 0).all())
            expected_passes = ((puzzles == 0).sum(dim=1) + 4) // 5
            self.assertTrue(torch.equal(passes, expected_passes))

    def test_one_shot_methods_agree_and_random_repeats(self):
        predictions = []
        for method in METHODS:
            result, _ = solve(ContextModel(), self.puzzles, method, cells_per_step=81)
            predictions.append(result)
        self.assertTrue(all(torch.equal(predictions[0], result) for result in predictions))

        first, _ = solve(ContextModel(), self.puzzles, "random", seed=42)
        second, _ = solve(ContextModel(), self.puzzles, "random", seed=42)
        self.assertTrue(torch.equal(first, second))

    def test_fifty_step_schedules_complete_with_equal_budgets(self):
        for schedule in ["linear", "binomial"]:
            counts = make_reveal_schedule(torch.tensor([1, 23, 61]), 50, schedule, 7)
            self.assertTrue(torch.equal(counts.sum(dim=1), torch.tensor([1, 23, 61])))
            self.assertTrue((counts >= 0).all())

            pass_counts = []
            for method in METHODS:
                predictions, passes = solve(
                    ContextModel(), self.puzzles, method,
                    seed=7, steps=50, gumbel_scale=.5, schedule=schedule,
                )
                self.assertTrue((predictions > 0).all())
                self.assertTrue(torch.equal(
                    predictions[self.puzzles != 0], self.puzzles[self.puzzles != 0],
                ))
                pass_counts.append(passes)
            self.assertTrue(all(torch.equal(pass_counts[0], passes) for passes in pass_counts))

    def test_native_numpy_loading_reconstructs_board_coordinates(self):
        # Original records use solving order, which is not necessarily board order.
        solution = self.solutions[0]
        cells = np.arange(81)[::-1]
        strategies = np.where(np.arange(81) < 24, 0, 2)
        moves = np.column_stack([cells // 9, cells % 9, solution[cells], strategies])
        records = np.concatenate([[24], moves.flatten()])[None, :]

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.npy"
            np.save(path, records)
            puzzles, solutions, ids, info = load_test_data(path, limit=0)
            np.testing.assert_array_equal(solutions[0], solution)
            np.testing.assert_array_equal(ids, [0])
            self.assertEqual(np.count_nonzero(puzzles[0]), 24)
            self.assertTrue(np.all(puzzles[0, :57] == 0))
            self.assertFalse(info["verified_original_test_set"])

        records[0, 5:7] = records[0, 1:3]  # Duplicate one coordinate.
        with self.assertRaises(ValueError):
            decode_records(records)

    def test_metrics_score_blanks_and_check_constraints(self):
        puzzles = self.puzzles.numpy()
        predictions = self.solutions.copy()
        metrics = evaluate(puzzles, self.solutions, predictions, np.array([50]), 1.0)
        self.assertEqual(metrics["exact_match_rate"], 1.0)
        self.assertEqual(metrics["blank_cell_accuracy"], 1.0)
        self.assertEqual(metrics["valid_rate"], 1.0)

        predictions[0, 2] = predictions[0, 0]  # Introduce a repeated row digit.
        metrics = evaluate(puzzles, self.solutions, predictions, np.array([50]), 1.0)
        self.assertEqual(metrics["valid_rate"], 0.0)
        self.assertEqual(metrics["exact_match_rate"], 0.0)
        self.assertTrue(valid_grid(self.solutions[0]))

    def test_wilson_boundary_cases(self):
        self.assertGreater(wilson_interval(0, 100)[1], 0)
        self.assertLess(wilson_interval(1, 100)[0], 1)

    def test_other_model_adapter_maps_mask_and_excludes_nondigit_class(self):
        class RecordingNetwork(torch.nn.Module):
            def forward(self, tokens):
                self.tokens = tokens.clone()
                return torch.arange(10).float().expand(*tokens.shape, 10)

        network = RecordingNetwork()
        adapter = DigitAdapter(network)
        board = self.puzzles.clone()
        original = board.clone()
        logits = adapter(board, torch.ones(1))
        self.assertTrue(torch.equal(board, original))
        self.assertTrue(torch.equal(network.tokens[board == 0], torch.full_like(board[board == 0], 10)))
        self.assertTrue(torch.equal(network.tokens[board > 0], board[board > 0]))
        self.assertEqual(logits.shape, (1, 81, 9))
        self.assertTrue(torch.equal(logits[0, 0], torch.arange(1, 10).float()))
        predictions, _ = solve(adapter, board, "probability", cells_per_step=81,
                               digit_mode="greedy")
        self.assertTrue((predictions[board == 0] == 9).all())

    def test_paired_intervals_use_puzzle_differences(self):
        self.assertEqual(paired_interval(np.zeros(20), repeats=100), (0., 0., 0.))
        effect, low, high = paired_interval(np.ones(20), repeats=100)
        self.assertEqual((effect, low, high), (1., 1., 1.))
        delta = np.array([-1., 0., 1., 1.])
        self.assertEqual(paired_interval(delta, repeats=100), paired_interval(delta, repeats=100))

    def test_cross_model_report_pairs_arrays_and_rejects_mismatched_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folders = [root / "harry-500k", root / "harry-hard"]
            settings = dict(sample_seed=42, seeds=[0, 1], steps=50, batch_size=32,
                            gumbel_scale=0, schedule="binomial", protocol="paper-inspired",
                            smoke_test=False)
            for index, folder in enumerate(folders):
                folder.mkdir()
                runs = []
                for method in METHODS:
                    for seed in settings["seeds"]:
                        filename = f"{method}_{seed}.npz"
                        prediction = self.solutions.copy()
                        # Only model b's probability/seed 1 prediction is wrong.
                        if index == 1 and method == "probability" and seed == 1:
                            prediction[0, 2] = 1
                        np.savez(folder / filename, puzzle_ids=[123], puzzles=self.puzzles.numpy(),
                                 solutions=self.solutions, predictions=prediction)
                        runs.append(dict(method=method, seed=seed, predictions_file=filename))
                summary = dict(settings=settings, dataset=dict(sha256="same"), runs=runs,
                               model=dict(name=folder.name, revision="test", parameters=1))
                (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            compare(folders, root / "report", baseline="harry-500k")
            report = json.loads((root / "report/comparison.json").read_text())
            # Seed repeats average inside the single puzzle; N remains 1.
            self.assertEqual(report["puzzles"], 1)
            model_b = report["models"][1]
            self.assertEqual(model_b["accuracy"]["probability"], .5)
            self.assertEqual(model_b["contrasts"][1]["change_from_baseline"], .5)
            training = report["same_architecture_comparisons"]
            self.assertEqual(training[0]["change"], -.5)
            self.assertEqual(training[1]["change"], .5)
            path = folders[1] / "probability_0.npz"
            np.savez(path, puzzle_ids=[999], puzzles=self.puzzles.numpy(),
                     solutions=self.solutions, predictions=self.solutions)
            with self.assertRaisesRegex(ValueError, "differ"):
                compare(folders, root / "bad_report", baseline="harry-500k")

    def test_noise_report_measures_paired_drop_and_checks_control(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folders = [root / "zero", root / "noisy"]
            for folder, scale in zip(folders, [0, .5]):
                folder.mkdir()
                runs = []
                for method in METHODS:
                    for seed in [0, 1]:
                        prediction = self.solutions.copy()
                        if scale and method == "probability" and seed == 1:
                            prediction[0, 2] = 1
                        filename = f"{method}_{seed}.npz"
                        np.savez(folder / filename, puzzle_ids=[123], puzzles=self.puzzles.numpy(),
                                 solutions=self.solutions, predictions=prediction)
                        metrics = evaluate(self.puzzles.numpy(), self.solutions, prediction,
                                           np.array([50]), 1.)
                        runs.append(dict(method=method, seed=seed, predictions_file=filename, **metrics))
                summary = dict(
                    model=dict(name="test", revision="fixed", parameters=1),
                    dataset=dict(sha256="same", selected_puzzle_ids=[123]), runs=runs,
                    settings=dict(sample_seed=42, seeds=[0, 1], steps=50, batch_size=32,
                                  gumbel_scale=scale, schedule="binomial", protocol="paper-inspired",
                                  smoke_test=False, device="cpu", torch_version=torch.__version__),
                )
                (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            report = compare_noise(folders, root / "report")
            probability = next(row for row in report["rows"]
                               if row["method"] == "probability" and row["gumbel_scale"] == .5)
            self.assertEqual(probability["exact_accuracy"], .5)
            self.assertEqual(probability["change_from_zero"], -.5)
            self.assertEqual(probability["change_ci_low"], -.5)
            self.assertEqual(probability["change_ci_high"], -.5)
            self.assertEqual(report["rows"][4]["change_from_zero"], 0.)
            self.assertTrue((root / "report/noise_comparison.png").read_bytes().startswith(b"\x89PNG"))
            self.assertTrue((root / "report/noise_comparison.pdf").read_bytes().startswith(b"%PDF"))
            path = folders[1] / "summary.json"
            saved_summary = path.read_text(encoding="utf-8")
            mismatch_mode = json.loads(saved_summary)
            mismatch_mode["settings"]["noise_mode"] = "upstream"
            path.write_text(json.dumps(mismatch_mode), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "noise_mode"):
                compare_noise(folders, root / "mixed_conventions")
            path.write_text(saved_summary, encoding="utf-8")
            mismatch_digits = json.loads(saved_summary)
            mismatch_digits["settings"]["digit_mode"] = "sample"
            path.write_text(json.dumps(mismatch_digits), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digit_mode"):
                compare_noise(folders, root / "mixed_digit_modes")
            path.write_text(saved_summary, encoding="utf-8")
            mismatch = json.loads(saved_summary)
            mismatch["settings"]["batch_size"] = 64
            path.write_text(json.dumps(mismatch), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "batch_size"):
                compare_noise(folders, root / "bad_report")
            path.write_text(saved_summary, encoding="utf-8")
            bad = self.solutions.copy()
            bad[0, 2] = 1
            np.savez(folders[1] / "random_0.npz", puzzle_ids=[123], puzzles=self.puzzles.numpy(),
                     solutions=self.solutions, predictions=bad)
            with self.assertRaisesRegex(ValueError, "control failed"):
                compare_noise(folders, root / "bad_report")


if __name__ == "__main__":
    unittest.main()
