from typing import Optional
from fifteen_puzzle_solvers.puzzle import Puzzle
from scipy.stats import norm
from sympy import Float

from FFNN import FFNN
from WUNN import WUNN

import random
import time
import torch


class PuzzleState:
    def __init__(self):
        self.state = Puzzle(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]
        )
        self.ffnn: Optional[FFNN] = None
        self.alpha: Optional[Float] = None
        self.y_q: Optional[Float] = None
        self.epsilon: Optional[Float] = None

    def set_params(self, **kwargs):
        self.ffnn = kwargs["ffnn"]
        self.alpha = kwargs["alpha"]
        self.y_q = kwargs["y_q"]
        self.epsilon = kwargs["epsilon"]

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
        if self.ffnn is None or self.alpha is None or self.y_q is None:
            raise Exception()

        mean, sigma_a_squared = self.ffnn.test(self.F(puzzle))
        mean = mean.item()

        if mean < self.y_q:
            sigma_t_squared = sigma_a_squared.item()
        else:
            sigma_t_squared = self.epsilon

        h_val = self.h(self.alpha, mean, sigma_t_squared).item()
        return max(0, h_val)

    def IDA_star(self, start_puzzle: Puzzle, t_max: int):
        start_time = time.time()
        bound = self.heuristic(start_puzzle)
        while True:
            result = self.search(start_puzzle, 0, bound, [start_puzzle])

            elapsed_time = time.time() - start_time
            if elapsed_time > t_max:
                return None

            if len(result["solution"]) > 0:
                return result
            if result["cost"] == float("inf"):
                return None
            bound = result["cost"]

    def search(self, puzzle: Puzzle, g, bound, path):
        if self.is_goal(puzzle):
            return {"cost": g, "solution": path}

        f = g + self.heuristic(puzzle)
        if f > bound:
            return {"cost": f, "solution": []}

        min = float("inf")
        for successor in puzzle.get_moves():
            if successor in path:
                continue
            new_path = path + [successor]
            result = self.search(successor, g + 1, bound, new_path)
            if len(result["solution"]) == 0 and result["cost"] < min:
                min = result["cost"]
            elif len(result["solution"]) > 0:
                return result
        return {"cost": min, "solution": []}

    def h(self, alpha, mu, sigma):
        return norm.ppf(alpha, loc=mu, scale=sigma)

    def F(self, puzzle: Puzzle):
        state = []
        for i in puzzle.position:
            for j in i:
                state.append(j)
        return torch.tensor(state).float().unsqueeze(0)

    def F_not_as_tensor(self, puzzle: Puzzle):
        state = []
        for i in puzzle.position:
            for j in i:
                state.append(j)
        return state

    def is_goal(self, puzzle: Puzzle):
        return puzzle.position == puzzle.PUZZLE_END_POSITION
