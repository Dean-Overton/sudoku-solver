from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

from .forms import UploadForm
from PIL import Image
import os
import time
from uploadapp.utils import (
    sudoku_img_2_array,
    Sudoku,
    sudoku_arr_superimpose_img
)
import numpy as np


def process_file(fs, file, file_path):
    # Python function to process the file
    # Replace this with your custom logic
    processed_filename = f"{file.name}_sudoku_solution.jpg"
    processed_file_path = os.path.join(fs.location, processed_filename)
    with Image.open(file_path) as img:
        # Convert PIL image to a NumPy array
        image_np = np.array(img)
        # Convert RGB to BGR (OpenCV uses BGR by default)
        image_bgr = image_np[:, :, ::-1]

        array = sudoku_img_2_array(image_bgr)
        array = np.matrix(array)
        sod = Sudoku(array.copy())
        sod.show()

        start_time = time.time()
        sod.solve()
        tts = time.time() - start_time

        print("Solved Sudoku:")
        sod.show()

        solution_image = sudoku_arr_superimpose_img(
            original_board=array,
            solution=sod.board,
            original_image=image_bgr
        )
        solution_PIL = Image.fromarray(solution_image)

        solution_PIL.save(processed_file_path)
    return {
        "file_name": processed_file_path,
        "time_to_solve": tts
    }


def upload_view(request):
    if request.method == 'POST' and request.FILES.get('file'):
        upload_form = UploadForm(request.POST, request.FILES)
        if upload_form.is_valid():
            file = request.FILES['file']
            fs = FileSystemStorage()
            file_path = fs.save(file.name, file)
            file_path = fs.path(file_path)

            # Solve the sudoku
            result = process_file(fs, file, file_path)

            return render(
                request,
                'uploadapp/result.html',
                {
                    'original_image_url': fs.url(file),
                    'processed_image_url': fs.url(result['file_name']),
                    'time_to_solve': result['time_to_solve']
                }
            )

    else:
        upload_form = UploadForm()

    return render(request, 'uploadapp/upload.html', {'form': upload_form})
