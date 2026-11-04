"""Download and read the original Sudoku test data. No CSV conversion."""

import argparse
import hashlib
import random
import urllib.request
from pathlib import Path

import numpy as np

DEFAULT_DATA = Path("data/raw/Sudoku-test-data.npy")
SOURCE_URL = (
    "https://drive.usercontent.google.com/download?"
    "id=1QMnUMkGcO_E_loAYFDmNo27dA3aryVcx&export=download&confirm=t"
)
SOURCE_SHA256 = "f13f537637ac64fb8f4cd5f5c6343a196472c21cd6e596428ed514b9e558ff3a"

EXAMPLE_PUZZLE = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
EXAMPLE_SOLUTION = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"


def file_checksum(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def download_test_data(path):
    """Keep the original .npy file locally; download it only when missing."""
    if path.exists():
        return

    print("Downloading the original Sudoku test set (about 33 MB)...", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".download")

    with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
        with temporary_path.open("wb") as file:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                file.write(chunk)

    if file_checksum(temporary_path) != SOURCE_SHA256:
        raise ValueError("The download does not match the original dataset checksum.")

    temporary_path.replace(path)


def valid_grid(board):
    """A completed Sudoku must contain digits 1..9 in every row, column and box."""
    grid = np.asarray(board).reshape(9, 9)
    required_digits = set(range(1, 10))

    for row in grid:
        if set(row) != required_digits:
            return False

    for column in grid.T:
        if set(column) != required_digits:
            return False

    for row in range(0, 9, 3):
        for column in range(0, 9, 3):
            box = grid[row:row + 3, column:column + 3]
            if set(box.flatten()) != required_digits:
                return False

    return True


def select_puzzle_ids(total, limit, seed):
    """Use the same seeded reservoir sample as the previous benchmark."""
    if limit == 0 or limit >= total:
        return np.arange(total)

    generator = random.Random(seed)
    selected_ids = list(range(limit))

    for puzzle_id in range(limit, total):
        replacement = generator.randrange(puzzle_id + 1)
        if replacement < limit:
            selected_ids[replacement] = puzzle_id

    return np.array(selected_ids)


def decode_records(records):
    """Turn the original move records into puzzle and solution arrays."""
    if records.ndim != 2 or records.shape[1] != 325:
        raise ValueError("Expected original Sudoku records with shape (N, 325).")
    if not np.issubdtype(records.dtype, np.integer):
        raise ValueError("Sudoku records must contain integers.")

    puzzles = np.zeros((len(records), 81), dtype=np.int64)
    solutions = np.zeros_like(puzzles)

    for puzzle_index, record in enumerate(records):
        number_of_clues = int(record[0])
        moves = record[1:].reshape(81, 4).astype(np.int64)
        rows, columns, digits, strategies = moves.T

        if not 1 <= number_of_clues < 81:
            raise ValueError("Each puzzle needs between 1 and 80 clues.")
        if np.any((rows < 0) | (rows > 8) | (columns < 0) | (columns > 8)):
            raise ValueError("Row and column coordinates must be between 0 and 8.")
        if np.any((digits < 1) | (digits > 9)):
            raise ValueError("Solution digits must be between 1 and 9.")

        # Moves are in solving order. Put each digit back at its board position.
        cell_positions = rows * 9 + columns
        if len(np.unique(cell_positions)) != 81:
            raise ValueError("Each cell must appear exactly once.")

        if np.any(strategies[:number_of_clues] != 0) or np.any(strategies[number_of_clues:] == 0):
            raise ValueError("Clue count and strategy annotations disagree.")

        solutions[puzzle_index, cell_positions] = digits

        # Only the first moves are clues. Everything else stays masked (zero).
        clue_positions = cell_positions[:number_of_clues]
        puzzles[puzzle_index, clue_positions] = digits[:number_of_clues]

        if not valid_grid(solutions[puzzle_index]):
            raise ValueError(f"Invalid reference solution at selected record {puzzle_index}.")

    return puzzles, solutions


def load_test_data(path=DEFAULT_DATA, limit=1000, sample_seed=42):
    """Return NumPy arrays plus original row IDs and source information."""
    if limit < 0:
        raise ValueError("limit must be zero (all puzzles) or positive.")

    download_test_data(path)
    checksum = file_checksum(path)
    records = np.load(path, mmap_mode="r", allow_pickle=False)
    if len(records) == 0:
        raise ValueError("The test dataset is empty.")

    puzzle_ids = select_puzzle_ids(len(records), limit, sample_seed)
    puzzles, solutions = decode_records(records[puzzle_ids])

    information = {
        "path": str(path),
        "sha256": checksum,
        "verified_original_test_set": checksum == SOURCE_SHA256,
        "total_puzzles": len(records),
        "selected_puzzle_ids": puzzle_ids.tolist(),
    }
    return puzzles, solutions, puzzle_ids, information


def load_example():
    puzzles = np.array([[int(digit) for digit in EXAMPLE_PUZZLE]])
    solutions = np.array([[int(digit) for digit in EXAMPLE_SOLUTION]])
    information = {"path": "bundled example", "verified_original_test_set": False}
    return puzzles, solutions, np.array([0]), information


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()

    puzzles, _, _, _ = load_test_data(args.data, limit=0)
    print(f"Validated {len(puzzles):,} puzzles in {args.data}. No intermediate file needed.")


if __name__ == "__main__":
    main()
