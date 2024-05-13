import numpy as np


# All data files should load from a class into arrays
# of sudoku puzzles and solutions
import os


def load_example_function():
    # self.puzzles=[...], self.solutions=[...]
    pass


class load_1_million_sudoku:
    def __init__(self, length=1000000):
        """Load 1 million sudoku puzzles and solutions from sudoku.csv."""
        print("Loading 1 million sudoku puzzles and solutions...")
        quizzes = np.zeros((length, 81), np.int32)
        solutions = np.zeros((length, 81), np.int32)
        for i, line in enumerate(open('data/sudoku.csv', 'r').read().splitlines()[1:length+1]):
            quiz, solution = line.split(",")
            for j, q_s in enumerate(zip(quiz, solution)):
                q, s = q_s
                quizzes[i, j] = q
                solutions[i, j] = s

            os.system('cls')
            print(f"Loaded {i+1} sudoku puzzles and solutions")

        self.quizzes = quizzes.reshape((-1, 9, 9))
        self.solutions = solutions.reshape((-1, 9, 9))
