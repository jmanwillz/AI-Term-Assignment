from fifteen_puzzle_solvers.algorithms import AStar

from scipy.stats import norm

from FFNN import FFNN
from WUNN import WUNN
from Puzzles import Puzzles
import math
import numpy as np


def h(alpha, mu, sigma):
    return norm.ppf(alpha, loc=mu, scale=sigma)


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


def generate_task_prac(**kwargs):
    puzzle = Puzzles()
    if kwargs["wunn"].is_trained:
        pass
        # return puzzle.generate_puzzle_uncert()
    else:
        return puzzle.generate_puzzle(num_steps=1)

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

    memory_buffer = dict()
    y_q = float("-inf")
    alpha = kwargs["alpha_0"]
    beta = kwargs["beta_0"]
    update_beta = True

    for n in range(kwargs["num_iter"]):
        num_solved = 0
        for i in range(kwargs["num_tasks_per_iter"]):
            task = generate_task_prac(
                wunn=wunn, max_steps=kwargs["max_steps"], epsilon=kwargs["epsilon"]
            )

            strategy = AStar(task)
            if strategy.start is not None:
                if strategy.start.is_solvable():
                    strategy.solve_puzzle()
                    plan = strategy.solution
                    num_solved += 1


if __name__ == "__main__":
    learn_heuristic_prac(
        num_iter=50,
        alpha_0=0.99,
        beta_0=0.05,
        num_tasks_per_iter=10,
        max_steps=1000,
        epsilon=1,
    )
