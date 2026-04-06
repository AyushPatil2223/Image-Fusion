"""
Download EMMA pretrained model (EMMA.pth) from GitHub.
Run: python -m emma.download_model
"""

import os
import sys
import urllib.request

MODEL_URL = "https://github.com/Zhaozixiang1228/MMIF-EMMA/raw/main/model/EMMA.pth"
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(_SCRIPT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "EMMA.pth")


def download():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.isfile(MODEL_PATH):
        print(f"EMMA model already exists: {MODEL_PATH}")
        return MODEL_PATH

    print(f"Downloading EMMA.pth from {MODEL_URL} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        if os.path.isfile(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
            print(f"Success! Model saved to: {MODEL_PATH}")
            return MODEL_PATH
        else:
            os.remove(MODEL_PATH)
            raise IOError("Downloaded file is too small or invalid.")
    except Exception as e:
        print(f"Download failed: {e}")
        print("\nManual download:")
        print(f"  1. Open: {MODEL_URL}")
        print(f"  2. Save as: {MODEL_PATH}")
        sys.exit(1)


if __name__ == "__main__":
    download()
