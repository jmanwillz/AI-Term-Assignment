from typing import Optional
from fifteen_puzzle_solvers.puzzle import Puzzle
from scipy.stats import norm
from sympy import Float

from FFNN import FFNN
from WUNN import WUNN

import random


class PuzzleState:
    def __init__(self):
        self.state = Puzzle(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]
        )
        self.ffnn: Optional[FFNN] = None
        self.alpha: Optional[Float] = None

    def set_ffnn(self, ffnn: FFNN):
        self.ffnn = ffnn

    def set_alpha(self, alpha):
        self.alpha = alpha

    def generate_puzzle(self, num_steps):
        start_state = self.state
        state = start_state
        for step in range(num_steps):
            moves = state.get_moves()
            move = random.choice(moves)
            state = move
        self.state = state

    def generate_puzzle_uncert(self, max_uncert: float, wunn: WUNN, max_steps):
        pass

    def heuristic(self, puzzle: Puzzle):
        # TODO: Add logic here
        if self.ffnn is None or self.alpha is None:
            raise Exception()

        mean, variance = self.ffnn.test(self.F(puzzle))
        h = self.h(self.alpha, mean, variance)
        return max(0, h[0])

    def iterative_deepening_a_star_search(self, start_puzzle: Puzzle):
        threshold = self.heuristic(start_puzzle)
        while True:
            result = self.search(start_puzzle, 0, threshold)
            if isinstance(result, tuple) and result[0].is_goal():
                return result
            threshold = result

    def search(self, node, g, threshold):
        f = g + self.heuristic(node)
        if f > threshold:
            return f
        if node.is_goal():
            return node, g
        min_threshold = float("inf")
        for neighbor in node.get_neighbors():
            result = self.search(neighbor, g + 1, threshold)
            if isinstance(result, tuple) and result[0].is_goal():
                return result
            elif isinstance(result, float):
                min_threshold = min(min_threshold, result)
        return min_threshold

    def h(self, alpha, mu, sigma):
        return norm.ppf(alpha, loc=mu, scale=sigma)

    def F(self, puzzle: Puzzle):
        pass
