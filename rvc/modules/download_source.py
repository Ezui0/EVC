import urllib.request
from urllib.parse import urlparse

import gradio as gr
import requests
from rvc.lib.download import mega, mediafire, pixeldrain


# Universal function to download a file from various sources
def download_file(url, zip_name, progress):
    try:
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname
        if hostname == "drive.google.com":
            download_from_google_drive(url, zip_name, progress)
        elif hostname == "huggingface.co":
            download_from_huggingface(url, zip_name, progress)
        elif hostname == "pixeldrain.com":
            download_from_pixeldrain(url, zip_name, progress)
        elif hostname == "mega.nz":
            download_from_mega(url, zip_name, progress)
        elif hostname in {"disk.yandex.ru", "yadi.sk"}:
            download_from_yandex(url, zip_name, progress)
        else:
            raise ValueError(f"Unsupported source: {url}")  # Handle unsupported links
    except Exception as e:
        # Handle any errors raised during download
        raise gr.Error(f"Download error: {str(e)}")


# Download a file from Google Drive using urllib (avoids extra gdown dependency)
def download_from_google_drive(url, zip_name, progress):
    progress(0.5, desc="[~] Downloading model from Google Drive...")
    file_id = url.split("file/d/")[1].split("/")[0] if "file/d/" in url else url.split("id=")[1].split("&")[0]  # Extract the file ID
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    urllib.request.urlretrieve(download_url, zip_name)


# Download a file from HuggingFace using urllib
def download_from_huggingface(url, zip_name, progress):
    progress(0.5, desc="[~] Downloading model from HuggingFace...")
    urllib.request.urlretrieve(url, zip_name)


# Download a file from Pixeldrain via its API
def download_from_pixeldrain(url, zip_name, progress):
    progress(0.5, desc="[~] Downloading model from Pixeldrain...")
    file_id = url.split("pixeldrain.com/u/")[1]  # Extract the file ID
    response = requests.get(f"https://pixeldrain.com/api/file/{file_id}")
    with open(zip_name, "wb") as f:
        f.write(response.content)


# Download a file from Mega using the mega module
# The mega module provides a standalone implementation that does not require the Mega SDK class.
# If the module fails to import or handle the URL, a helpful error is raised.
def download_from_mega(url, zip_name, progress):
    progress(0.5, desc="[~] Downloading model from Mega...")
    try:
        import os
        dest_dir = os.path.dirname(zip_name)
        mega.mega_download_url(url, dest_path=dest_dir if dest_dir else ".")
    except Exception as e:
        raise gr.Error(f"Failed to download from Mega: {str(e)}. Download the model via a direct link (e.g. HuggingFace).")


# Download a file from Yandex Disk via its public API
def download_from_yandex(url, zip_name, progress):
    progress(0.5, desc="[~] Downloading model from Yandex Disk...")
    yandex_public_key = f"download?public_key={url}"  # Build the public-key query parameter
    yandex_api_url = f"https://cloud-api.yandex.net/v1/disk/public/resources/{yandex_public_key}"
    response = requests.get(yandex_api_url)
    if response.status_code == 200:
        download_link = response.json().get("href")  # Get the download link
        urllib.request.urlretrieve(download_link, zip_name)
    else:
        # Handle error while fetching the Yandex Disk download link
        raise gr.Error(f"Error getting download link from Yandex Disk: {response.status_code}")
