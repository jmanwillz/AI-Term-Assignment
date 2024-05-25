from os import path
from fifteen_puzzle_solvers.puzzle import Puzzle

from Helper import (
    generate_puzzles,
    get_optimal_plans,
    read_puzzles_from_file,
    save_puzzles_to_file,
)


if __name__ == "__main__":
    # puzzles = generate_puzzles(100)
    # save_puzzles_to_file(puzzles)
    puzzles: list[Puzzle] = read_puzzles_from_file(
        file_path="/Users/jmanwillz/Desktop/AI Term Assignment/output/puzzles_2024-05-25_11-43-01.json"
    )

    optimal_plans = get_optimal_plans(puzzles)
