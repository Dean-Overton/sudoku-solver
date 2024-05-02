import util
import cv2
import matplotlib.pyplot as plt
import pytesseract


def get_preprocessed_img_from(image: cv2.imread):
    """This function preprocesses an image to extract the sudoku puzzle."""

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (3, 3), 6)

    # Apply adaptive thresholding to binarize the image
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    return thresh


def sudoku_img_2_array(image):
    """This function extracts the sudoku puzzle from the image and converts it\
    to a 2D array."""
    image = get_preprocessed_img_from(image)

    # Display the result
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    print("Reading the Sudoku puzzle image...")
    image = cv2.imread('./test-data/puzzle3.jpg')
    array = sudoku_img_2_array(image)
    sod = util.Sudoku(array)

    print("Solving the Sudoku puzzle...")
    sod.solve()

    print("Solved Sudoku:")
    sod.show()


if __name__ == "__main__":
    main()
