from fifteen_puzzle_solvers.puzzle import Puzzle
from queue import Queue
from typing import Optional

from FFNN import FFNN
from FifteenPuzzle import FifteenPuzzle
from Helper import F_as_list, generate_task_prac, is_goal
from WUNN import WUNN

import math
import torch


def learn_heuristic_prac(**kwargs):
    wunn = WUNN(mu_0=kwargs["mu_0"], sigma_0=math.sqrt(kwargs["sigma_0_squared"]))
    ffnn = FFNN()

    memory_buffer = Queue()
    y_q = float("-inf")
    alpha = kwargs["alpha_0"]
    beta = kwargs["beta_0"]
    update_beta = True

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

        train_set = list(memory_buffer.queue)
        x = []
        y = []
        for entry in train_set:
            x.append(F_as_list(entry[0]))
            y.append(entry[1])

        ffnn.train(
            torch.tensor(x).float(), torch.tensor(y).float(), kwargs["train_iter"]
        )
        wunn.train(
            torch.tensor(x).float(), torch.tensor(y).float(), kwargs["max_train_iter"]
        )

        print()
        print(f"Finished iteration: {n + 1}/{kwargs['num_iter']}")
        print(f"Memory buffer size: {memory_buffer.qsize()}")
        print()

    return wunn, ffnn


if __name__ == "__main__":
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
