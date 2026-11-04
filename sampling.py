"""The experiment: choose which masked cells to fill at each step."""

import torch

METHODS = ("random", "probability", "margin", "entropy")


def score_cells(probabilities, method, generator):
    """Give every cell one score. Higher means fill this cell sooner."""
    if method == "random":
        return torch.rand(probabilities.shape[:2], generator=generator).to(probabilities.device)

    if method == "probability":
        return probabilities.max(dim=-1).values

    if method == "margin":
        best_two = probabilities.topk(2, dim=-1).values
        return best_two[:, :, 0] - best_two[:, :, 1]

    if method == "entropy":
        log_probabilities = probabilities.clamp_min(1e-12).log()
        negative_entropy = (probabilities * log_probabilities).sum(dim=-1)
        return negative_entropy

    raise ValueError(f"Unknown ordering method: {method}")


def make_reveal_schedule(initial_blanks, steps, schedule, seed):
    """Precompute shared reveal counts so every method gets the same budget."""
    initial_blanks = initial_blanks.cpu().long()
    remaining_blanks = initial_blanks.clone()
    generator = torch.Generator().manual_seed(seed)
    counts = []

    for step in range(steps):
        if schedule == "binomial":
            # Last step has probability 1, so every puzzle finishes.
            reveal_probability = 1 / (steps - step)
            probabilities = torch.full_like(remaining_blanks.float(), reveal_probability)
            count = torch.binomial(remaining_blanks.float(), probabilities, generator=generator).long()

        elif schedule == "linear":
            target_filled = (initial_blanks * (step + 1) + steps - 1) // steps
            already_filled = initial_blanks - remaining_blanks
            count = target_filled - already_filled

        else:
            raise ValueError(f"Unknown reveal schedule: {schedule}")

        counts.append(count)
        remaining_blanks -= count

    return torch.stack(counts, dim=1)


def add_ordering_noise(scores, scale, generator):
    """Optional Gumbel noise on raw scores, using the previous convention."""
    uniform = torch.rand(scores.shape, generator=generator).to(scores.device)
    uniform = uniform.clamp(1e-7, 1 - 1e-7)
    gumbel_noise = -torch.log(-torch.log(uniform))
    return scores + scale * gumbel_noise


def choose_cells(scores, masked_cells, reveal_counts):
    """Select the highest-scoring masked cells in each puzzle."""
    scores = scores.masked_fill(~masked_cells, float("-inf"))
    sorted_cells = scores.argsort(dim=1, descending=True, stable=True)
    ranks = torch.arange(scores.shape[1], device=scores.device).expand_as(sorted_cells)
    selected_ranks = ranks < reveal_counts.to(scores.device).unsqueeze(1)
    selected_cells = torch.zeros_like(masked_cells).scatter(1, sorted_cells, selected_ranks)
    return selected_cells & masked_cells


@torch.inference_mode()
def solve(model, puzzles, method, *, seed=0, steps=50, cells_per_step=None,
          schedule="binomial", gumbel_scale=0.0):
    """Repeatedly predict digits, choose cells, and fill those cells."""
    if method not in METHODS:
        raise ValueError(f"Unknown ordering method: {method}")
    if steps < 1 or (cells_per_step is not None and not 1 <= cells_per_step <= 81):
        raise ValueError("steps must be positive and cells_per_step must be between 1 and 81.")
    if not 0 <= gumbel_scale < float("inf"):
        raise ValueError("gumbel_scale must be finite and nonnegative.")
    if puzzles.ndim != 2 or puzzles.shape[1] != 81 or len(puzzles) == 0:
        raise ValueError("puzzles must have shape (nonempty batch, 81 cells).")

    board = puzzles.clone()
    initial_blanks = (board == 0).sum(dim=1)
    passes = torch.zeros(len(board), dtype=torch.long, device=board.device)
    generator = torch.Generator().manual_seed(seed)

    if cells_per_step is None:
        total_steps = steps
        reveal_counts = make_reveal_schedule(initial_blanks, steps, schedule, seed + 1000003)
    else:
        maximum_blanks = int(initial_blanks.max().item())
        total_steps = (maximum_blanks + cells_per_step - 1) // cells_per_step
        reveal_counts = torch.full((len(board), total_steps), cells_per_step)

    for step in range(total_steps):
        masked_cells = board == 0
        unfinished_puzzles = masked_cells.any(dim=1)
        if not unfinished_puzzles.any():
            break

        if cells_per_step is None:
            timestep = torch.full((len(board),), 1 - step / steps, device=board.device)
        else:
            timestep = masked_cells.sum(dim=1).float() / initial_blanks.clamp_min(1)

        # 1. Predict probabilities for digits 1..9 at every cell.
        logits = model(board, timestep)
        if logits.shape != (*board.shape, 9):
            raise ValueError("The model must return (batch size, 81 cells, 9 digits).")
        probabilities = logits.float().softmax(dim=-1)
        predicted_digits = probabilities.argmax(dim=-1) + 1

        # 2. Choose which cells to fill. This is the only difference between methods.
        scores = score_cells(probabilities, method, generator)
        if method != "random" and gumbel_scale > 0:
            scores = add_ordering_noise(scores, gumbel_scale, generator)
        selected_cells = choose_cells(scores, masked_cells, reveal_counts[:, step])

        # 3. Fill selected blanks. Clues and earlier predictions stay fixed.
        board[selected_cells] = predicted_digits[selected_cells]
        passes += unfinished_puzzles.long()

    return board, passes
