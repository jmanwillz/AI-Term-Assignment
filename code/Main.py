from fifteen_puzzle_solvers.puzzle import Puzzle
from queue import Queue
from typing import Optional

from FFNN import FFNN
from FifteenPuzzle import FifteenPuzzle
from Helper import (
    generate_puzzles,
    generate_task_prac,
    get_gamma,
    get_optimal_plans,
    get_x_and_y_for_training,
    is_goal,
)
from WUNN import WUNN

import math
import numpy as np
import torch


def learn_heuristic_prac(**kwargs):
    update_beta = True
    beta = kwargs["beta_0"]
    gamma = get_gamma(beta_0=beta, beta_num_iter=0.00001, num_iter=kwargs["num_iter"])
    wunn = WUNN(
        mu_0=kwargs["mu_0"],
        sigma_0=math.sqrt(kwargs["sigma_0_squared"]),
        update_beta=update_beta,
        beta=beta,
        gamma=gamma,
    )
    ffnn = FFNN()

    memory_buffer = Queue()
    y_q = float("-inf")
    alpha = kwargs["alpha_0"]

    for n in range(kwargs["num_iter"]):
        num_solved = 0
        for i in range(kwargs["num_tasks_per_iter"]):
            T: Optional[Puzzle] = generate_task_prac(
                wunn=wunn,
                max_steps=kwargs["max_steps"],
                K=kwargs["K"],
                epsilon=kwargs["epsilon"],
            )

            if T is None or is_goal(T):
                continue

            puzzle: FifteenPuzzle = FifteenPuzzle(
                alpha=alpha,
                check_time=True,
                epsilon=kwargs["epsilon"],
                ffnn=ffnn,
                puzzle=T,
                t_max=kwargs["t_max"],
                uncertain=True,
                y_q=y_q,
            )

            plan = puzzle.ida_star()

            if plan is None:
                continue

            num_solved += 1
            cost = plan["cost"]
            for s_j in plan["solution"]:
                if not is_goal(s_j):
                    x_j = s_j
                    y_j = cost
                    cost -= 1

                    if memory_buffer.qsize() > kwargs["max_memory_buffer_records"]:
                        memory_buffer.get()
                    memory_buffer.put((x_j, y_j))

        if num_solved < kwargs["num_tasks_per_iter_thresh"]:
            alpha = max(0.5, alpha - kwargs["delta"])
            update_beta = False
        else:
            update_beta = True

        x, y = get_x_and_y_for_training(list(memory_buffer.queue))

        ffnn.train(
            torch.tensor(x).float(), torch.tensor(y).float(), kwargs["train_iter"]
        )
        wunn.train(
            torch.tensor(x).float(), torch.tensor(y).float(), kwargs["max_train_iter"]
        )

        y_q = np.quantile(y, kwargs["q"])

        print()
        print(f"Finished iteration: {n + 1}/{kwargs['num_iter']}")
        print(f"Memory buffer size: {memory_buffer.qsize()}")
        print()

    return wunn, ffnn


if __name__ == "__main__":
    puzzles = generate_puzzles(100)
    optimal_plans = get_optimal_plans(puzzles)

    wunn, ffnn = learn_heuristic_prac(
        alpha_0=0.99,
        beta_0=0.05,
        delta=0.05,
        epsilon=1,
        K=100,
        max_memory_buffer_records=25000,
        max_steps=1000,
        max_train_iter=5000,
        mu_0=0,
        num_iter=50,
        num_tasks_per_iter=10,
        num_tasks_per_iter_thresh=6,
        q=0.95,
        sigma_0_squared=10,
        t_max=60,
        train_iter=1000,
    )
