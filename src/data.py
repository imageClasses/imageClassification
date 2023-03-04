"Some functions to handle data operations"
from urllib import request
import zipfile
import os

def fetch_data(folder="data", zip_file="file.zip", remove_zip=True, overwrite=False):
    if not overwrite:
        #Check if resulting folders already exists, skip fetching if does
        directories = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
        has_annotations = False
        has_images = False
        for d in directories:
            if d == "annotations":
                has_annotations = True
            if d == "images":
                has_images = True
        
        if has_annotations and has_images:
            print("Folder already has folders 'annotations' and 'images'.")
            print("Assuming you already have the data and skipping fetch.")
            return None

    
    download_data(folder, zip_file)
    unzip_data(folder, zip_file, remove_zip)


def download_data(folder="data", zip_file="file.zip"):
    try: os.mkdir(folder)
    except: pass
    file_path = f"{folder}/{zip_file}"
    url = "https://www.cs.helsinki.fi/u/yangarbe/Courses/2023-deep-learning/image-training-corpus+annotations/dl2021-image-corpus-proj.zip"

    print("Downloading data...")
    request.urlretrieve(url, file_path)
    print("Downloading done")
    return file_path

def unzip_data(folder="data", zip_file="file.zip", remove_zip=True):
    try: os.mkdir(folder)
    except: pass
    file_path = f"{folder}/{zip_file}"
    
    print("Unzipping data...")
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(f"{folder}/")
    
    if remove_zip:
        os.remove(file_path)
    print("Unzipping done...")
