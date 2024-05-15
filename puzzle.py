import random


class Puzzle:
    # Custom implementation of the 15 puzzle.
    solution_state: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]

    def __init__(self) -> None:
        self.state = []

    @staticmethod
    def validate_puzzle(state: list[int]) -> bool:
        if len(state) != 16:
            return False
        for value in range(16):
            if value not in state:
                return False

        return True

    @staticmethod
    def hamming(state: list[int]) -> int:
        if not Puzzle.validate_puzzle(state):
            raise ValueError("Invalid puzzle.")
        result: int = 0
        for index in range(16):
            if state[index] != 0:
                if state[index] != index + 1:
                    result += 1
        return result

    def hamming(self) -> int:
        return Puzzle.hamming(self.state)

    def set_state(self, values: list[int]):
        if not Puzzle.validate_puzzle(values):
            raise ValueError("Invalid puzzle.")
        self.state = values

    def initialise_state(self):
        elements = [x for x in range(16)]
        self.state = random.sample(elements, len(elements))

    def get_index_from_row_and_column(self, row: int, column: int) -> int:
        return (row * 4) + column

    def move_up(self):
        if self.get_row() == 0:
            return

        index_of_zero = self.get_index_from_row_and_column(
            self.get_row(), self.get_column()
        )
        index_of_value_above = self.get_index_from_row_and_column(
            self.get_row() - 1, self.get_column()
        )
        self.state[index_of_zero] = self.state[index_of_value_above]
        self.state[index_of_value_above] = 0

    def move_down(self):
        if self.get_row() == 3:
            return

        index_of_zero = self.get_index_from_row_and_column(
            self.get_row(), self.get_column()
        )
        index_of_value_below = self.get_index_from_row_and_column(
            self.get_row() + 1, self.get_column()
        )
        self.state[index_of_zero] = self.state[index_of_value_below]
        self.state[index_of_value_below] = 0

    def move_left(self):
        if self.get_column() == 0:
            return

        index_of_zero = self.get_index_from_row_and_column(
            self.get_row(), self.get_column()
        )
        index_of_value_left = self.get_index_from_row_and_column(
            self.get_row(), self.get_column() - 1
        )
        self.state[index_of_zero] = self.state[index_of_value_left]
        self.state[index_of_value_left] = 0

    def move_right(self):
        if self.get_column() == 3:
            return

        index_of_zero = self.get_index_from_row_and_column(
            self.get_row(), self.get_column()
        )
        index_of_value_right = self.get_index_from_row_and_column(
            self.get_row(), self.get_column() + 1
        )
        self.state[index_of_zero] = self.state[index_of_value_right]
        self.state[index_of_value_right] = 0

    def move(self, direction: str):
        if direction.lower() == "left":
            self.move_left()
        elif direction.lower() == "right":
            self.move_right()
        elif direction.lower() == "up":
            self.move_up()
        elif direction.lower() == "down":
            self.move_down()
        else:
            raise ValueError(f"The direction {direction} is not valid.")

    def get_row(self, value=0):
        index = self.state.index(value)
        return (index / 4).__floor__()

    def get_column(self, value=0):
        index = self.state.index(value)
        return index % 4

    def __repr__(self) -> str:
        result = []
        counter = 0
        for _ in range(4):
            row = []
            for _ in range(4):
                row.append(self.state[counter])
                counter += 1
            result.append(row)
        return str(result)

    def __str__(self) -> str:
        result = ""
        counter = 0
        for column in range(4):
            for _ in range(4):
                if self.state[counter] == 0:
                    result += "X\t"
                else:
                    result += f"{self.state[counter]}\t"
                counter += 1
            result = result.strip()
            if column < 3:
                result += "\n"
        return result
