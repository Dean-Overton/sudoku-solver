from data.load_sudoku_algo_test_data import load_1_million_sudoku
import util
import numpy as np
import os


def main():
    test_size = int(input("How many puzzles do you want to test?"))

    data = load_1_million_sudoku(length=test_size)
    x = data.quizzes
    y = data.solutions

    correct_solutions = 0
    incorrect_solutions = 0
    for index in range(test_size):
        this_puzzle = util.Sudoku(x[index])

        solution = this_puzzle.solve()

        if np.array_equal(solution, y[index]):
            correct_solutions += 1
        else:
            incorrect_solutions += 1

        # TODO: MAC support
        os.system('cls')
        print(f"""Solved: {index+1},\n
solutions correct:{correct_solutions}/{index + 1},\n
running accuracy: {correct_solutions / (index + 1) * 100:.2f}%
              """)


if __name__ == "__main__":
    main()
