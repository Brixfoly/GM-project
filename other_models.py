"""Adapt the linked SudokuDiffusion checkpoints to our common digit-only decoder.

Architecture transcribed from diffusion_baseline.py in the public repository;
never import its module, which executes dataset loading at module scope.
"""

import hashlib
from pathlib import Path
from urllib.request import urlopen

import torch
from torch import nn

REPOSITORY = "harry-david-brown/sudoku-diffusive-masked"
REVISION = "ed3730064ded2e31fab8c65185435c15241be07e"
CHECKPOINTS = {
    "harry-100k": ("sudoku_diffusion_masked_100k.pth", "07702ea1428fee79ffca0005839771ab368aeca49276f14a5de1fa4d01553eb0"),
    "harry-500k": ("sudoku_diffusion_masked_500k.pth", "9c6a1b42411496cb84a4801e712f53d9477b6c5ed00c187573a1c75437520a35"),
    "harry-combined": ("sudoku_diffusion_masked_combined.pth", "7d7610b43c312a88d569578ca4aabfceef22b1c0f1b933253b282cb28942a0d6"),
    "harry-hard": ("sudoku_diffusion_masked_finetuned_hard.pth", "a1cbb0bf731012abcb480ac2ae82612f1d951c026c9a059869c116fbf48fa0c8"),
}


class SudokuDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(11, 128)
        self.pos_embedding = nn.Embedding(81, 128)
        layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=512,
            dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=4)
        self.output = nn.Linear(128, 10)

    def forward(self, tokens):
        positions = torch.arange(81, device=tokens.device).unsqueeze(0)
        hidden = self.embedding(tokens) + self.pos_embedding(positions)
        return self.output(self.transformer(hidden))


class DigitAdapter(nn.Module):
    """Map our blank=0 to MASK=10 and restrict outputs to legal digits 1..9.

The upstream network has no timestep input. Its unused class 0 is excluded
before softmax, so every ordering rule uses the same nine-digit distribution
as the original DiT. This is a controlled decoder, not the upstream sampler.
"""
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, board, timestep):
        tokens = board.masked_fill(board == 0, 10)
        return self.network(tokens)[..., 1:10]


def load_other_model(name, device, cache=Path("data/models")):
    filename, expected_hash = CHECKPOINTS[name]
    folder = cache / "harry-david-brown" / REVISION
    path = folder / filename
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/{filename}"
        with urlopen(url, timeout=60) as response:
            content = response.read()
        path.write_bytes(content)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_hash is not None and checksum != expected_hash:
        raise ValueError(f"Checkpoint checksum mismatch: {path}")
    network = SudokuDiffusion()
    network.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    model = DigitAdapter(network).to(device).eval()
    return model, {
        "name": name, "repository": f"https://github.com/{REPOSITORY}",
        "revision": REVISION, "checkpoint": filename, "sha256": checksum,
        "parameters": sum(p.numel() for p in model.parameters()),
        "adapter": "blank 0 -> mask 10; softmax over digit logits 1..9; no timestep",
    }
