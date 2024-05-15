from puzzle import *
from globals import *
import os


def train(
    run: int,
    iterations: int,
    num_output_solve: int,
    gen_domain_from_uncertain: bool,
    length_inc: int,
    eff_run: bool,
):
    run_type: str = get_run_type(
        num_output_solve, gen_domain_from_uncertain, length_inc, eff_run
    )
    csv_write_path: str = os.path.join(
        Globals.run_path, Globals.domain_name, run, run_type, Globals.train_file_name
    )

    num_hidden_solve = 20

    # if mult_heuristic != None:
    #     num_hidden_solve = 8


def main():
    for run_number in range(5):
        parameters = [(1, True, 1)]

        for parameter in parameters:
            train(
                run_number,
                Globals.number_iterations,
                parameter[0],
                parameter[1],
                parameter[2],
                False,
            )


def get_run_type(
    num_output_solve, gen_domain_from_uncertain, length_inc, eff_run
) -> str:
    if num_output_solve == 1 and gen_domain_from_uncertain and not eff_run:
        return "1"
    elif num_output_solve > 1 and gen_domain_from_uncertain:
        return "2"
    elif num_output_solve == 1 and gen_domain_from_uncertain and eff_run:
        return "1e"
    else:
        return "k" + str(length_inc)


if __name__ == "__main__":
    puzzle = Puzzle()
    puzzle.initialise_state()
    print(puzzle)
    column = puzzle.get_column()
    row = puzzle.get_row()
    print("Here")
