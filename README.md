# GM-project
Codebase to benchmark MDMs in sudoku using different token ordering strategies

## Dependencies :
PyTorch, huggingface_hub and what is in requirements.txt

## Usage :
Executing benchmark.py will test the MDM on sudoku grids with different inference strategies:
- random
- probability
- margin
- lowest entropy
    and report the accuracy of each one.
The detailed results are in results/benchmark and are compared with the results in the paper.