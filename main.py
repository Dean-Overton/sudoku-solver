import util
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image, ImageEnhance


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


def reframe(points):
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
    raw_image_copy = raw_image.copy()

    biggest, maxArea = main_outline(contour)

    if biggest.size != 0:
        biggest = reframe(biggest)
        cv2.drawContours(raw_image_copy, biggest, -1, (0, 255, 0), 10)
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
        enhanced_img = enhancer.enhance(3.0)

        # Convert the image to grayscale
        bw_img = enhanced_img.convert('L')

        Cells_croped.append(bw_img)

    return Cells_croped


def sudoku_img_2_array(raw_image):
    """This function extracts the sudoku puzzle from the image and converts it\
    to a 2D array."""
    raw_image = cv2.resize(raw_image, (450, 450))

    processed_image = get_preprocessed_img_from(raw_image)

    contour, hiarachy = get_contours_from(
        raw_image, processed_image, debug=False)

    su_cropped = get_sudoku_cropped_image_from(raw_image, contour)

    sudoku_cells = get_cell_array_from_img(su_cropped)
    sudoku_cells_cropped = get_cropped_cells_from_array(sudoku_cells)

    # Extract the numbers from the cells
    su_arr = []
    for row in range(0, 9):
        row_arr = []
        for col in range(0, 9):
            row_arr.append(pytesseract.image_to_string(
                sudoku_cells_cropped[row * 9 + col],
                lang='eng',
                config='--psm 10 --oem 3 -c \
                tessedit_char_whitelist=0123456789')
            )

        su_arr.append(row_arr)

    # Clean the array
    for i in range(0, 9):
        for j in range(0, 9):
            if su_arr[i][j] == '':
                su_arr[i][j] = 0
            else:
                su_arr[i][j] = int(su_arr[i][j])

    return su_arr


def main():
    print("Reading the Sudoku puzzle image...")
    image = cv2.imread('./test-data/puzzle2.jpg')
    array = sudoku_img_2_array(image)
    array = np.matrix(array)
    print(array)
    # sod = util.Sudoku(array)

    print("Solving the Sudoku puzzle...")
    # sod.solve()

    print("Solved Sudoku:")
    # sod.show()


if __name__ == "__main__":
    main()
