from fifteen_puzzle_solvers.puzzle import Puzzle

from WUNN import WUNN

import random


class Puzzles:
    def __init__(self):
        pass

    def generate_puzzle(self, num_steps) -> Puzzle:
        start_state = Puzzle(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]
        )
        state = start_state
        for step in range(num_steps):
            moves = state.get_moves()
            move = random.choice(moves)
            state = move
        return state

    def generate_puzzle_uncert(self, max_uncert: float, wunn: WUNN, max_steps):
        pass
