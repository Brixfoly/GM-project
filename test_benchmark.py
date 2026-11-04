"""Small offline checks for the data reader and solving experiment."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from prepare_data import EXAMPLE_SOLUTION, decode_records, load_example, load_test_data, valid_grid
from reporting import evaluate, wilson_interval
from sampling import METHODS, choose_cells, make_reveal_schedule, score_cells, solve


class ContextModel(torch.nn.Module):
    """A fake model whose predictions change when the board changes."""
    def forward(self, board, timestep):
        digits = (torch.arange(81)[None, :] + board.sum(dim=1, keepdim=True)) % 9
        return torch.nn.functional.one_hot(digits, 9).float() * 3


class BenchmarkTests(unittest.TestCase):
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
                            dict(gumbel_scale=-1), dict(gumbel_scale=float("nan"))]
        for settings in invalid_settings:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                solve(ContextModel(), self.puzzles, "margin", **settings)

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


if __name__ == "__main__":
    unittest.main()
