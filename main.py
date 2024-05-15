from puzzle import *

if __name__ == "__main__":
    puzzle = Puzzle()
    puzzle.initialise_state()
    print(puzzle)
    column = puzzle.get_column()
    row = puzzle.get_row()
    print("Here")
