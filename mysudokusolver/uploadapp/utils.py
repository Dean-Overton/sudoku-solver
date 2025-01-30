import cv2
import time
from PIL import Image, ImageEnhance
import pytesseract
import matplotlib.pyplot as plt
import numpy as np


class Sudoku:
    def __init__(self, board: np.matrix = None):
        if board is not None:
            self.board = board
        else:
            self.board = np.matrix([[0 for i in range(9)] for j in range(9)])

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

        return True

    def _is_safe(self, grid_x, grid_y, row, col, num):
        """Check if it is safe to place the number in the cell"""
        # Runtime complexity: O(N^2) where N is the size of the board (9x9)
        return not self._check_row_for_number(row, num) and \
            not self._check_col_for_number(col, num) and \
            not self._check_matrix_for_number(num, grid_x, grid_y)

    def backtracking_solve(self, row=0, column=0):
        """Backtracking algorithm that recursively solves a soduku starting
        branching from values placed from top left to bottom right."""

        if row == 8 and column == 9:
            # If we reach the end of the board, we have solved the puzzle
            return True

        if column == 9:
            # Move to next row if at the end of the column
            row += 1
            column = 0

        if self.board[row, column] > 0:
            # Skip filled cells
            return self.backtracking_solve(row, column + 1)

        grid_x = column // 3
        grid_y = row // 3

        for num in range(1, 10):
            # Place values that are safe/valid only
            if self._is_safe(grid_x, grid_y, row, column, num):
                self.board[row, column] = num

                # Try solving the next cell/cells after placeing this value
                if self.backtracking_solve(row, column + 1):
                    return True

                self.board[row, column] = 0

        return False

    def solve(self):
        # Backtracking algorithm to solve the sudoku
        self.backtracking_solve()

        # FINAL CHECK
        result = self._check_sudoku()
        print(f"Result can solve sudoku: {result}")
        return self.board

    def show(self):
        for i in range(9):
            for j in range(8):
                print(self.board[i, j], end=" |")
            print(self.board[i, 8])


def get_preprocessed_img_from(image: cv2.imread):
    """This function preprocesses an image to extract the sudoku puzzle."""

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (3, 3), 6)

    # Apply adaptive thresholding to binarize the image
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    return thresh


def main_outline(contour):
    biggest = np.array([])
    max_area = 0
    for i in contour:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)

            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

    return biggest, max_area


def get_cell_array_from_img(img):
    rows = np.vsplit(img, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)

        for box in cols:
            boxes.append(box)
    return boxes


def reframe(points: np.ndarray):
    points = points.reshape((4, 2))
    points_new = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)
    points_new[0] = points[np.argmin(add)]
    points_new[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    points_new[1] = points[np.argmin(diff)]
    points_new[2] = points[np.argmax(diff)]
    return points_new


def get_contours_from(raw_image, image, debug=False):
    """This function extracts contours from the image."""
    contour_1 = raw_image.copy()
    contour, hierarchy = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_1, contour, -1, (0, 255, 0), 3)

    if debug:
        plt.figure()
        plt.imshow(contour_1)
        plt.show()

    return contour, hierarchy


def get_sudoku_cropped_image_from(raw_image, contour):
    biggest, maxArea = main_outline(contour)

    if biggest.size != 0:
        biggest = reframe(biggest)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imagewrap = cv2.warpPerspective(raw_image, matrix, (450, 450))
        imagewrap = cv2.cvtColor(imagewrap, cv2.COLOR_BGR2GRAY)
        return imagewrap


def get_cropped_cells_from_array(cells):
    Cells_croped = []
    for image in cells:
        img = np.array(image)
        img = img[4:46, 6:46]
        img = Image.fromarray(img)
        enhancer = ImageEnhance.Contrast(img)
        # Increase the contrast by a factor of 2
        enhanced_img = enhancer.enhance(2.0)

        # Convert the image to grayscale
        bw_img = enhanced_img.convert('L')

        Cells_croped.append(bw_img)

    return Cells_croped


def sudoku_img_2_array(raw_image):
    """This function extracts the sudoku puzzle from the image and converts it\
    to a 2D array."""
    start_time = time.time()
    raw_image = cv2.resize(raw_image, (450, 450))

    processed_image = get_preprocessed_img_from(raw_image)

    contour, hiarachy = get_contours_from(
        raw_image, processed_image, debug=False)

    su_cropped = get_sudoku_cropped_image_from(raw_image, contour)

    sudoku_cells = get_cell_array_from_img(su_cropped)
    sudoku_cells_cropped = get_cropped_cells_from_array(sudoku_cells)

    # Show specific cell
    # img = sudoku_cells_cropped[45]
    # img.show()

    # Extract the numbers from the cells
    su_arr = []
    for row in range(0, 9):
        row_arr = []
        for col in range(0, 9):
            # PSM 10: Find single character. Can miss some characters Eg 9.

            # PSM 6: Assume a single uniform block of text. Can miss interpret
            # but picks up more values

            # Oem 3: Default OCR Engine Mode.

            row_arr.append(pytesseract.image_to_string(
                sudoku_cells_cropped[row * 9 + col],
                lang='eng',
                config='--psm 10 --oem 3 -c \
                tessedit_char_whitelist=0123456789')
            )

        su_arr.append(row_arr)

    print(f"Time taken to detect sudoku: {time.time() - start_time}")

    # Clean the array
    for i in range(0, 9):
        for j in range(0, 9):
            if su_arr[i][j] == '':
                su_arr[i][j] = 0
            else:
                su_arr[i][j] = int(su_arr[i][j])

    return su_arr


def sudoku_arr_superimpose_img(
    original_board: np.matrix,
    solution: np.matrix,
    original_image: np.ndarray
):
    """This function superimposes the solution on the original image."""
    original_image = cv2.resize(original_image, (450, 450))

    # Clean solution array by removing original values
    print("Og Board:")
    print(original_board)
    for i in range(9):
        for j in range(9):
            if original_board[i, j] != 0:
                solution[i, j] = 0

    print("Solution:")
    print(solution)

    # Create solution image
    solution_image = np.zeros((450, 450, 4), dtype=np.uint8)
    for i in range(9):
        for j in range(9):
            if solution[i, j] != 0:
                cv2.putText(
                    solution_image,
                    str(solution[i, j]),
                    (j * 50 + 20, i * 50 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255, 255),
                    2
                )

    processed_image = get_preprocessed_img_from(original_image)

    contour, hiarachy = get_contours_from(
        original_image, processed_image, debug=False)

    biggest, maxArea = main_outline(contour)

    if biggest.size != 0:
        biggest = reframe(biggest)
        pts1 = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])
        pts2 = np.float32(biggest)
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imagewrap = cv2.warpPerspective(solution_image, matrix, (450, 450))

        if imagewrap.shape[2] == 4:
            b, g, r, alpha = cv2.split(imagewrap)
        else:
            raise ValueError("Foreground image must have an alpha channel!")

    # Resize foreground to fit background if necessary
    imagewrap = cv2.resize(
        imagewrap, (original_image.shape[1], original_image.shape[0]))
    alpha = cv2.resize(
        alpha, (original_image.shape[1], original_image.shape[0]))

    # Normalize alpha mask to range [0, 1]
    alpha = alpha.astype(float) / 255.0

    # Convert to 3-channel mask
    alpha = cv2.merge([alpha, alpha, alpha])

    # Blend images using alpha mask
    foreground_bgr = cv2.merge([b, g, r])  # Remove alpha channel
    blended = (foreground_bgr * alpha + original_image *
               (1 - alpha)).astype(np.uint8)

    cv2.imshow("Blended", foreground_bgr)
    cv2.imshow("Original", original_image)
    cv2.imshow("Solution", imagewrap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return blended
