from queue import Queue

from FFNN import FFNN
from PuzzleState import PuzzleState
from WUNN import WUNN

import math
import numpy as np
import torch


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def sample_from_states(states):
    state_values = list(states.values())
    state_keys = list(states.keys())
    probabilities = softmax(np.array(state_values))
    sampled_index = np.random.choice(len(state_keys), p=probabilities)
    sampled_state = state_keys[sampled_index]
    sampled_value = state_values[sampled_index]
    return sampled_state, sampled_value


def generate_task_prac(**kwargs) -> PuzzleState:
    puzzle = PuzzleState()
    if kwargs["wunn"].is_trained:
        raise Exception()
        # return puzzle.generate_puzzle_uncert()
    else:
        puzzle.generate_puzzle(num_steps=1)
        return puzzle

    # s_prime = Puzzle([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]])
    # num_steps = 0
    # s_double_prime = None
    # states = dict()

    # while num_steps < kwargs["max_steps"]:
    #     num_steps += 1
    #     states = dict()
    #     for s in s_prime.get_moves():
    #         if s_double_prime is not None and s_double_prime != s:
    #             continue

    #         x = s.position
    #         tensor_x = torch.unsqueeze(
    #             torch.tensor(list(np.concatenate(x).flat), dtype=torch.float32), dim=0
    #         )
    #         sigma_e_squared = kwargs["wunn"].test(tensor_x)
    #         print(f"Value: {sigma_e_squared}")
    #         states[s] = math.sqrt(sigma_e_squared)

    #     s, sigma_e = sample_from_states(states)

    #     if sigma_e**2 >= kwargs["epsilon"]:
    #         return s

    #     s_double_prime = s_prime
    #     s_prime = s


def learn_heuristic_prac(**kwargs):
    wunn = WUNN(mu_0=0, sigma_0=math.sqrt(10))
    ffnn = FFNN()

    memory_buffer = Queue()
    y_q = float("-inf")
    alpha = kwargs["alpha_0"]
    beta = kwargs["beta_0"]
    update_beta = True

    for n in range(kwargs["num_iter"]):
        num_solved = 0
        for i in range(kwargs["num_tasks_per_iter"]):
            puzzle_state = generate_task_prac(
                wunn=wunn,
                epsilon=kwargs["epsilon"],
                max_steps=kwargs["max_steps"],
                K=kwargs["K"],
            )

            puzzle_state.set_params(
                alpha=alpha, ffnn=ffnn, y_q=y_q, epsilon=kwargs["epsilon"]
            )

            plan = puzzle_state.IDA_star(puzzle_state.state, t_max=kwargs["t_max"])

            if plan is not None:
                num_solved += 1
                for s_j in plan["solution"]:
                    if not puzzle_state.is_goal(s_j):
                        x_j = s_j
                        y_j = plan["cost"]

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
            x.append(puzzle_state.F_not_as_tensor(entry[0]))
            y.append(entry[1])

        ffnn.train(
            torch.tensor(x).float(), torch.tensor(y).float(), kwargs["train_iter"]
        )
        wunn.train(
            torch.tensor(x).float(), torch.tensor(y).float(), kwargs["max_train_iter"]
        )


if __name__ == "__main__":
    learn_heuristic_prac(
        alpha_0=0.99,
        beta_0=0.05,
        delta=0.05,
        epsilon=1,
        K=100,
        max_memory_buffer_records=25000,
        max_steps=1000,
        max_train_iter=5000,
        num_iter=50,
        num_tasks_per_iter=10,
        num_tasks_per_iter_thresh=6,
        q=0.95,
        t_max=60,
        train_iter=1000,
    )
