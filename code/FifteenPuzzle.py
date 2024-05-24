import math
from typing import Optional
from fifteen_puzzle_solvers.puzzle import Puzzle
import time

from FFNN import FFNN
from Helper import F, h, is_goal, manhattan_distance


class FifteenPuzzle:
    def __init__(
        self,
        alpha: float,
        check_time: bool,
        epsilon: float,
        ffnn: FFNN,
        puzzle: Puzzle,
        t_max: int,
        uncertain: bool,
        y_q: float,
    ) -> None:
        self.alpha: float = alpha
        self.check_time: bool = check_time
        self.epsilon: float = epsilon
        self.ffnn: FFNN = ffnn
        self.puzzle: Puzzle = puzzle
        self.t_max: int = t_max
        self.uncertain: bool = uncertain
        self.y_q: float = y_q

    def heuristic(self, puzzle: Puzzle) -> float:
        if not self.uncertain:
            return manhattan_distance(puzzle)
        else:
            return self.our_heuristic(puzzle)

    def our_heuristic(self, puzzle: Puzzle) -> float:
        mean, variance = self.ffnn.test(F(puzzle))

        if mean < self.y_q:
            sigma_t_squared = variance
        else:
            sigma_t_squared = self.epsilon

        h_val = h(alpha=self.alpha, mu=mean, sigma=math.sqrt(sigma_t_squared)).item()
        return max(0, h_val)

    def ida_star(self) -> Optional[dict]:
        start_puzzle = self.puzzle
        start_time = time.time()
        bound = self.heuristic(puzzle=start_puzzle)
        while True:
            result = self.__search(
                puzzle=start_puzzle,
                g=0,
                bound=bound,
                path=[start_puzzle],
                start_time=start_time,
            )

            elapsed_time = time.time() - start_time
            if elapsed_time > self.t_max and self.check_time:
                return None

            if len(result["solution"]) > 0:
                return result
            if result["cost"] == float("inf"):
                return None
            bound = result["cost"]

    def __search(
        self, bound: float, g: int, path: list[Puzzle], puzzle: Puzzle, start_time
    ):
        if is_goal(puzzle):
            return {"cost": g, "solution": path}

        f = g + self.heuristic(puzzle=puzzle)
        if f > bound:
            return {"cost": f, "solution": []}

        min = float("inf")
        for successor in puzzle.get_moves():
            if successor in path:
                continue
            new_path = path + [successor]
            result = self.__search(
                puzzle=successor,
                g=g + 1,
                bound=bound,
                path=new_path,
                start_time=start_time,
            )

            elapsed_time = time.time() - start_time
            if elapsed_time > self.t_max and self.check_time:
                return {"cost": -1, "solution": []}

            if len(result["solution"]) == 0 and result["cost"] == -1:
                return result
            elif len(result["solution"]) == 0 and result["cost"] < min:
                min = result["cost"]
            elif len(result["solution"]) > 0:
                return result
        return {"cost": min, "solution": []}
