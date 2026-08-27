import os

import requests
from tqdm import tqdm

PREDICTORS = "https://huggingface.co/Politrees/RVC_resources/resolve/main/predictors/"
EMBEDDERS = "https://huggingface.co/Politrees/RVC_resources/resolve/main/embedders/pytorch/"

PREDICTORS_DIR = os.path.join(os.getcwd(), "rvc", "models", "predictors")
EMBEDDERS_DIR = os.path.join(os.getcwd(), "rvc", "models", "embedders")

# Create folders if they do not exist
os.makedirs(PREDICTORS_DIR, exist_ok=True)
os.makedirs(EMBEDDERS_DIR, exist_ok=True)


def dl_model(link, model_name, dir_name):
    file_path = os.path.join(dir_name, model_name)
    if os.path.exists(file_path):
        return  # Skip download if the file already exists

    try:
        r = requests.get(f"{link}{model_name}", stream=True, timeout=(10, 300))
        r.raise_for_status()

        # Get the total file size
        total_size = int(r.headers.get("content-length", 0))
        # Use tqdm to display progress
        with open(file_path, "wb") as f, tqdm(
            desc=f"Downloading {model_name}",
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
    except Exception:
        # Remove the incomplete file so the next attempt starts fresh
        if os.path.exists(file_path):
            os.remove(file_path)
        raise


def check_and_install_models(offline=False):
    try:
        if offline:
            print("Running in OFFLINE mode - skipping model downloads.")
            return

        predictors_names = ["rmvpe.pt", "fcpe.pt"]
        for model in predictors_names:
            dl_model(PREDICTORS, model, PREDICTORS_DIR)

        embedder_names = ["hubert_base.pt"]
        for model in embedder_names:
            dl_model(EMBEDDERS, model, EMBEDDERS_DIR)

    except requests.exceptions.RequestException as e:
        print(f"Error downloading model: {e}")
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
