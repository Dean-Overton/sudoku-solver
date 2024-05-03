import numpy as np


class Sudoku:
    def __init__(self, board: np.matrix = None):
        if board is not None:
            self.board = board
        else:
            self.board = np.matrix([[0 for i in range(9)] for j in range(9)])

        print("Sudoku created:")
        print(self.board)

    def _check_row_for_number(self, row, num):
        """Helper function: check if number is in row"""
        for col in range(9):
            if self.board[row, col] == num:
                return True
        return False

    def _check_col_for_number(self, col, num):
        """Helper function: check if number is in column"""
        for row in range(9):
            if self.board[row, col] == num:
                return True
        return False

    def _check_matrix_for_number(
            self,
            search: int,
            x_matrix: int,
            y_matrix: int):
        """Helper function: check if number is in a 3x3 sudoku matrix"""

        for row in range(y_matrix*3, (y_matrix*3) + 3):
            for col in range(x_matrix*3, (x_matrix*3) + 3):
                if self.board[row, col] == search:
                    return True

        return False

    def _check_sudoku(self):
        """Check if the sudoku board is valid"""
        for num in range(1, 10):
            for row in range(9):
                if not self._check_row_for_number(row, num):
                    return False

            for col in range(9):
                if not self._check_col_for_number(col, num):
                    return False

            for x_matrix in range(3):
                for y_matrix in range(3):
                    if not self._check_matrix_for_number(
                            num,
                            x_matrix,
                            y_matrix):
                        return False

    def _is_safe(self, grid_x, grid_y, row, col, num):
        """Check if it is safe to place the number in the cell"""
        # Runtime complexity: O(N^2) where N is the size of the board (9x9)
        return not self._check_row_for_number(row, num) and \
            not self._check_col_for_number(col, num) and \
            not self._check_matrix_for_number(num, grid_x, grid_y)

    def solve(self):
        # TODO: Implement the Sudoku puzzle solving algorithm

        result = self._check_sudoku()
        print(f"Result can solve sudoku: {result}")
        # if result:
        print(self.board)
        return self.board

    def show(self):
        for i in range(9):
            for j in range(8):
                print(self.board[i, j], end=" |")
            print(self.board[i, 8])
