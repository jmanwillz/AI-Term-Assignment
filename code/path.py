from fifteen_puzzle_solvers.algorithms import AStar, BreadthFirst
from fifteen_puzzle_solvers.puzzle import Puzzle
from fifteen_puzzle_solvers.solver import PuzzleSolver
from scipy.stats import norm

import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn


class WUNN:
    def __init__(self, mu_0, sigma_0):
        self.is_trained = False
        self.model = nn.Sequential(
            bnn.BayesLinear(
                prior_mu=mu_0, prior_sigma=sigma_0, in_features=16, out_features=20
            ),
            nn.ReLU(),
            bnn.BayesLinear(
                prior_mu=mu_0, prior_sigma=sigma_0, in_features=20, out_features=1
            ),
        )

        self.mse_loss = nn.MSELoss()
        self.kl_loss = bnn.BKLLoss(reduction="mean", last_layer_only=False)
        self.kl_weight = 0.01
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def train(self, x, y):
        num_epochs = 3000
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            outputs = self.model(x)

            mse = self.mse_loss(outputs, y)
            kl_loss = self.kl_loss(self.model)
            loss = mse + self.kl_weight * kl_loss

            loss.backward()
            self.optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, KL Loss: {kl_loss.item():.4f}"
                )
        self.is_trained = True

    def test(self, x_test):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(x_test)
        return predictions


class FFNN:
    def __init__(self):
        self.is_trained = False
        self.model = nn.Sequential(nn.Linear(16, 20), nn.ReLU(), nn.Linear(20, 2))
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def negative_log_likelihood(self, outputs, targets):
        mean = outputs[:, 0]
        log_variance = outputs[:, 1]
        variance = torch.exp(log_variance)
        nll = 0.5 * torch.mean(log_variance + ((targets - mean) ** 2) / variance)
        return nll

    def train(self, x, y):
        num_epochs = 3000
        self.model.train()
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.negative_log_likelihood(outputs, y)
            loss.backward()
            self.optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        self.is_trained = True

    def test(self, x_test):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x_test)
            mean = outputs[:, 0]
            log_variance = outputs[:, 1]
            variance = torch.exp(log_variance)
        return mean, variance


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
