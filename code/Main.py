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
    alpha = kwargs["alpha_0"]
    beta = kwargs["beta_0"]
    beta_num_iter = kwargs["beta_num_iter"]
    delta = kwargs["delta"]
    epsilon = kwargs["epsilon"]
    K = kwargs["K"]
    max_memory_buffer_records = kwargs["max_memory_buffer_records"]
    max_steps = kwargs["max_steps"]
    max_train_iter = kwargs["max_train_iter"]
    mu_0 = kwargs["mu_0"]
    num_iter = kwargs["num_iter"]
    num_tasks_per_iter = kwargs["num_tasks_per_iter"]
    num_tasks_per_iter_thresh = kwargs["num_tasks_per_iter_thresh"]
    q = kwargs["q"]
    sigma_0_squared = kwargs["sigma_0_squared"]
    t_max = kwargs["t_max"]
    train_iter = kwargs["train_iter"]

    memory_buffer = Queue()
    gamma = get_gamma(beta_0=beta, beta_num_iter=beta_num_iter, num_iter=num_iter)
    update_beta = True
    y_q = float("-inf")

    wunn = WUNN(
        mu_0=mu_0,
        sigma_0=math.sqrt(sigma_0_squared),
        update_beta=update_beta,
        beta=beta,
        gamma=gamma,
    )
    ffnn = FFNN()

    for n in range(num_iter):
        num_solved = 0
        for _ in range(num_tasks_per_iter):
            T: Optional[Puzzle] = generate_task_prac(
                wunn=wunn,
                max_steps=max_steps,
                K=K,
                epsilon=epsilon,
            )

            if T is None or is_goal(T):
                continue

            puzzle: FifteenPuzzle = FifteenPuzzle(
                alpha=alpha,
                check_time=True,
                epsilon=epsilon,
                ffnn=ffnn,
                puzzle=T,
                t_max=t_max,
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

                    if memory_buffer.qsize() > max_memory_buffer_records:
                        memory_buffer.get()
                    memory_buffer.put((x_j, y_j))

        if num_solved < num_tasks_per_iter_thresh:
            alpha = max(0.5, alpha - delta)
            update_beta = False
        else:
            update_beta = True

        x, y = get_x_and_y_for_training(list(memory_buffer.queue))

        ffnn.train(torch.tensor(x).float(), torch.tensor(y).float(), train_iter)
        wunn.train(torch.tensor(x).float(), torch.tensor(y).float(), max_train_iter)

        y_q = np.quantile(y, q)

        print()
        print(f"Finished iteration: {n + 1}/{num_iter}")
        print(f"Memory buffer size: {memory_buffer.qsize()}")
        print()

    return wunn, ffnn


if __name__ == "__main__":
    wunn, ffnn = learn_heuristic_prac(
        alpha_0=0.99,
        beta_0=0.05,
        beta_num_iter=0.00001,
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
