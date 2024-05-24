from typing import Optional
from fifteen_puzzle_solvers.puzzle import Puzzle
from scipy.stats import norm

from FFNN import FFNN
from WUNN import WUNN

import math
import numpy as np
import random
import time
import torch


class PuzzleState:
    def __init__(self):
        self.state = Puzzle(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]
        )
        self.ffnn: Optional[FFNN] = None
        self.alpha: Optional[float] = None
        self.y_q: Optional[float] = None
        self.epsilon: Optional[float] = None

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

    def generate_puzzle_uncert(
        self, wunn: WUNN, epsilon: float, max_steps: int, K: int
    ):
        visited_states = set()
        s_prime: Puzzle = Puzzle(self.state.PUZZLE_END_POSITION)
        visited_states.add(s_prime)
        num_steps = 0

        while num_steps < max_steps:
            num_steps += 1
            states = dict()
            for s in s_prime.get_moves():
                if s in visited_states:
                    continue

                x = self.F(s)
                results = []
                for sample in range(K):
                    results.append(wunn.test(x).item())
                variance = np.var(results)
                states[s] = math.sqrt(variance)
                visited_states.add(s)

            key = random.choice(list(states.keys()))
            sample = states[key]

            if sample**2 >= epsilon:
                self.state = key
                return

            s_prime = s

    def manhattan_distance(self, puzzle: Puzzle):
        goal_positions = {
            1: (0, 0),
            2: (0, 1),
            3: (0, 2),
            4: (0, 3),
            5: (1, 0),
            6: (1, 1),
            7: (1, 2),
            8: (1, 3),
            9: (2, 0),
            10: (2, 1),
            11: (2, 2),
            12: (2, 3),
            13: (3, 0),
            14: (3, 1),
            15: (3, 2),
            0: (3, 3),
        }

        total_distance = 0

        for i in range(4):
            for j in range(4):
                tile = puzzle.position[i][j]
                if tile != 0:
                    goal_i, goal_j = goal_positions[tile]
                    total_distance += abs(i - goal_i) + abs(j - goal_j)

        return total_distance

    def heuristic(self, puzzle: Puzzle, uncert: bool = True):
        if not uncert:
            return self.manhattan_distance(puzzle)
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

    def IDA_star(
        self,
        start_puzzle: Puzzle,
        t_max: int,
        uncert: bool = True,
        check_time: bool = True,
    ):
        start_time = time.time()
        bound = self.heuristic(start_puzzle, uncert)
        while True:
            result = self.search(start_puzzle, 0, bound, [start_puzzle], uncert)

            elapsed_time = time.time() - start_time
            if elapsed_time > t_max and check_time:
                return None

            if len(result["solution"]) > 0:
                return result
            if result["cost"] == float("inf"):
                return None
            bound = result["cost"]

    def search(self, puzzle: Puzzle, g, bound, path, uncert: bool = True):
        if self.is_goal(puzzle):
            return {"cost": g, "solution": path}

        f = g + self.heuristic(puzzle, uncert)
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

    def is_goal(self, puzzle: Optional[Puzzle] = None):
        if puzzle is not None:
            return puzzle.position == puzzle.PUZZLE_END_POSITION
        else:
            return self.state.position == self.state.PUZZLE_END_POSITION

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
