import numpy as np


class Sudoku:
    def __init__(self, board: np.matrix = None):
        if board is not None:
            self.board = board
        else:
            self.board = np.matrix([[0 for i in range(9)] for j in range(9)])

        print("Sudoku created:")
        print(self.board)

    def solve(self):
        # TODO: Implement the Sudoku puzzle solving algorithm
        pass

    def show(self):
        for i in range(9):
            for j in range(8):
                print(self.board[i, j], end=" |")
            print(self.board[i, 8])
