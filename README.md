# Sudoku Photo Solver

This problem was part of a Modern AI group project where we had to solve a sudoku puzzle from a photo. I decided to solve this problem using python and the OpenCV library with an added DJango App to provide a nice looking web interface for the algorithm.

The program should:

- automatically recognise a specific sudoku puzzle from the given image and extract relevant information.
- Use an algorithm to automatically solve the recognized sudoku puzzle.
- Display the solution to the recognized sudoku puzzle

### To run locally

1. Clone the repository
2. Install the required libraries using the following command:
   `pip install -r requirements.txt`
3. Change to the Django app directory:
   `cd mysudokusolver`
4. Run the server and visit the website locally:
  `python manage.py runserver`

### Testing

In order to test the program, you should download a test dataset in the `Data/` folder. This will NOT be pushed to the github.
[1 Million Sudoku Dataset](https://www.kaggle.com/datasets/bryanpark/sudoku?resource=download)
