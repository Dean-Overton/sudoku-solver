from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .forms import UploadForm


def process_file(file_path):
    # Python function to process the file
    # Replace this with your custom logic
    with open(file_path, mode='rb') as f:
        data = f.read()
    return f"File processed with {len(data)} characters."


def upload_view(request):
    if request.method == 'POST' and request.FILES.get('file'):
        upload_form = UploadForm(request.POST, request.FILES)
        if upload_form.is_valid():
            file = request.FILES['file']
            fs = FileSystemStorage()
            file_path = fs.save(file.name, file)
            file_path = fs.path(file_path)

            # Run Python function
            result = process_file(file_path)

            return render(request, 'uploadapp/result.html', {'result': result})

    else:
        upload_form = UploadForm()

    return render(request, 'uploadapp/upload.html', {'form': upload_form})
