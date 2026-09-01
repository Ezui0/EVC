import os
import shutil
import sys
import zipfile

import gradio as gr

from rvc.modules.download_source import download_file

# Path to the directory where RVC models are stored
rvc_models_dir = os.path.join(os.getcwd(), "models", "RVC_models")
os.makedirs(rvc_models_dir, exist_ok=True)


# Extracts a zip archive into the given directory and locates the model files (.pth and .index)
def extract_zip(extraction_folder, zip_name):
    os.makedirs(extraction_folder, exist_ok=True)  # Create the extraction directory if it does not exist
    with zipfile.ZipFile(zip_name, "r") as zip_ref:
        zip_ref.extractall(extraction_folder)  # Extract the zip archive
    os.remove(zip_name)  # Remove the zip archive after extraction

    index_filepath, model_filepath = None, None
    # Walk through all files in the extracted directory looking for .pth and .index
    for root, _, files in os.walk(extraction_folder):
        for name in files:
            file_path = os.path.join(root, name)
            if name.endswith(".index") and os.stat(file_path).st_size > 1024 * 100:  # Minimum index file size
                index_filepath = file_path
            if name.endswith(".pth") and os.stat(file_path).st_size > 1024 * 1024 * 40:  # Minimum pth file size
                model_filepath = file_path

    if not model_filepath:
        # Raise an error if no model file was found
        raise gr.Error("No .pth model file found in the extracted zip archive. Check the contents of {extraction_folder}.")

    # Rename files and remove unneeded folders
    rename_and_cleanup(extraction_folder, model_filepath, index_filepath)


# Renames files and removes empty folders
def rename_and_cleanup(extraction_folder, model_filepath, index_filepath):
    os.rename(
        model_filepath,
        os.path.join(extraction_folder, os.path.basename(model_filepath)),
    )
    if index_filepath:
        os.rename(
            index_filepath,
            os.path.join(extraction_folder, os.path.basename(index_filepath)),
        )

    # Remove leftover empty directories after extraction
    for filepath in os.listdir(extraction_folder):
        full_path = os.path.join(extraction_folder, filepath)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)


# Downloads a model from a URL and extracts the zip archive
def download_from_url(url, dir_name, progress=gr.Progress()):
    try:
        progress(0, desc=f"[~] Downloading voice model {dir_name}...")
        zip_name = os.path.join(rvc_models_dir, dir_name + ".zip")
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            # Check whether a directory with this name already exists
            raise gr.Error(f"Voice model directory {dir_name} already exists! Choose a different name for your voice model.")

        download_file(url, zip_name, progress)  # Download the file
        progress(0.8, desc="[~] Extracting zip archive...")
        extract_zip(extraction_folder, zip_name)  # Extract the zip archive
        return f"[+] Model {dir_name} successfully installed!"
    except Exception as e:
        # Handle errors during model installation
        raise gr.Error(f"Model installation error: {str(e)}")


# Handles uploading and extracting a model zip archive via the interface
def upload_zip_file(zip_path, dir_name, progress=gr.Progress()):
    try:
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(f"Voice model directory {dir_name} already exists! Choose a different name for your voice model.")

        zip_name = zip_path.name
        progress(0.8, desc="[~] Extracting zip archive...")
        extract_zip(extraction_folder, zip_name)  # Extract the zip archive
        return f"[+] Model {dir_name} successfully installed!"
    except Exception as e:
        # Handle errors during upload and extraction
        raise gr.Error(f"Model installation error: {str(e)}")


# Handles uploading separate model files (.pth and .index)
def upload_separate_files(pth_file, index_file, dir_name, progress=gr.Progress()):
    try:
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(f"Voice model directory {dir_name} already exists! Choose a different name for your voice model.")

        os.makedirs(extraction_folder, exist_ok=True)

        # Upload the .pth file
        progress(0.4, desc="[~] Uploading .pth file...")
        if pth_file:
            pth_path = os.path.join(extraction_folder, os.path.basename(pth_file.name))
            shutil.copyfile(pth_file.name, pth_path)

        # Upload the .index file
        progress(0.8, desc="[~] Uploading .index file...")
        if index_file:
            index_path = os.path.join(extraction_folder, os.path.basename(index_file.name))
            shutil.copyfile(index_file.name, index_path)

        return f"[+] Model {dir_name} successfully installed!"
    except Exception as e:
        # Handle errors during file upload
        raise gr.Error(f"Model installation error: {str(e)}")


# Entry point for command-line usage
def main():
    if len(sys.argv) != 3:
        print('\nUsage:\npython3 -m rvc.modules.model_manager "url" "dir_name"\n')
        sys.exit(1)

    url = sys.argv[1]
    dir_name = sys.argv[2]

    try:
        # Download and install the model from the command line
        result = download_from_url(url, dir_name)
        print(result)
    except gr.Error as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
