from datetime import datetime
from fifteen_puzzle_solvers.algorithms import AStar
from fifteen_puzzle_solvers.puzzle import Puzzle
from fifteen_puzzle_solvers.solver import PuzzleSolver
from scipy.stats import norm
from typing import Optional, Set

from WUNN import WUNN

import json
import math
import numpy as np
import random
import torch


def h(alpha, mu, sigma) -> float:
    return norm.ppf(alpha, loc=mu, scale=sigma).item()


def is_goal(puzzle: Puzzle) -> bool:
    return puzzle.position == puzzle.PUZZLE_END_POSITION


def F(s: Puzzle) -> torch.Tensor:
    state = []
    for row in s.position:
        for col in row:
            state.append(col)
    return torch.tensor(state).float().unsqueeze(0)


def get_gamma(beta_0: float, beta_num_iter: float, num_iter: int) -> float:
    return math.log(beta_num_iter / beta_0) / math.log(num_iter)


def F_as_list(puzzle: Puzzle) -> list:
    return list(np.array(puzzle.position).flatten())


def get_x_and_y_for_training(training_set: list):
    x = []
    y = []
    for entry in training_set:
        x.append(F_as_list(entry[0]))
        y.append(entry[1])
    return x, y


def get_optimal_plans(puzzles: list[Puzzle]):
    file_name = f"optimal_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    print("Solving optimal plans...")
    print()

    plans = dict()
    for index, puzzle in enumerate(puzzles):
        print(f"Solving puzzle {index + 1}/{len(puzzles)}")

        puzzle_solver = PuzzleSolver(AStar(puzzle))
        puzzle_solver.run()

        solution = []
        for step in puzzle_solver._strategy.solution:
            solution.append(step.position)

        plan = {
            "solution": solution,
            "cost": len(puzzle_solver._strategy.solution),
            "num_expanded_nodes": puzzle_solver._strategy.num_expanded_nodes,
        }

        plans[str(F_as_list(puzzle))] = plan

        print("Saving plan...")
        with open(file_name, "w") as outfile:
            json.dump(plans, outfile)
        print()

    return plans


def save_puzzles_to_file(puzzles: list[Puzzle]):
    results = {"puzzles": []}
    for puzzle in puzzles:
        results["puzzles"].append(puzzle.position)
    json_object = json.dumps(results, indent=4)
    file_name = f"puzzles_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    with open(file_name, "w") as outfile:
        outfile.write(json_object)


def read_puzzles_from_file(file_path: str) -> list[Puzzle]:
    with open(file_path, "r") as openfile:
        json_object = json.load(openfile)
    result: list[Puzzle] = []
    for puzzle in json_object["puzzles"]:
        result.append(Puzzle(puzzle))
    return result


def generate_puzzles(num_puzzles: int) -> list[Puzzle]:
    puzzles = []
    while len(puzzles) < num_puzzles:
        puzzle: Puzzle = Puzzle(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]
        )
        puzzle.generate_random_position()
        if puzzle in puzzles or not puzzle.is_solvable():
            continue
        puzzles.append(puzzle)
    return puzzles


def manhattan_distance(puzzle: Puzzle) -> int:
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
    for row in range(4):
        for col in range(4):
            tile = puzzle.position[row][col]
            if tile != 0:
                goal_row, goal_col = goal_positions[tile]
                total_distance += abs(row - goal_row) + abs(col - goal_col)
    return total_distance


def hamming(puzzle: Puzzle) -> int:
    hamming_distance = 0
    board = list(np.array(puzzle.position).flatten())
    for index, square in enumerate(board):
        if square != 0 and board[index] != index + 1:
            hamming_distance += 1
    return hamming_distance


def generate_task_prac(
    epsilon: float, K: int, max_steps: int, wunn: WUNN
) -> Optional[Puzzle]:
    puzzle: Optional[Puzzle] = None
    if wunn.is_trained:
        puzzle = generate_task_uncert(
            wunn=wunn,
            max_steps=max_steps,
            K=K,
            epsilon=epsilon,
        )
    else:
        puzzle = generate_task_cert(num_steps=1)
    return puzzle


def generate_task_uncert(
    epsilon: float, K: int, max_steps: int, wunn: WUNN
) -> Optional[Puzzle]:
    s_prime: Puzzle = Puzzle(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]
    )
    visited_states: Set = {s_prime}
    num_steps = 0

    while num_steps < max_steps:
        num_steps += 1
        states = dict()
        for s in s_prime.get_moves():
            if s in visited_states:
                continue

            x = F(s)
            predictions = []
            for _ in range(K):
                predictions.append(wunn.test(x))

            variance = np.var(predictions)
            states[s] = variance
            visited_states.add(s)

        s = random.choice(list(states.keys()))
        variance = states[s]

        if variance >= epsilon:
            return s

        s_prime = s

    return None


def generate_task_cert(num_steps: int) -> Puzzle:
    start_state: Puzzle = Puzzle(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]
    )
    visited_states: Set = {start_state}
    state: Puzzle = start_state

    for _ in range(num_steps):
        prev_states = state.get_moves()
        prev_state = random.choice(prev_states)
        prev_states.remove(prev_state)

        while len(prev_states) > 0 and prev_state in visited_states:
            prev_state = random.choice(prev_states)
            prev_states.remove(prev_state)
        if len(prev_states) == 0 and prev_state in visited_states:
            break

        state = prev_state
        visited_states.add(state)

    if is_goal(state):
        raise Exception("The returned puzzle should not be the goal state.")
    else:
        return state
