import os
import numpy as np
import cv2

# 🗂️ Base directory where your HR, LR, and IR folders are stored
BASE_DIR = r"C:\Users\ayush\Downloads\alignimages\preprocessed_ordered"
OUTPUT_DIR = os.path.join(BASE_DIR, "resized_128")

# Create output folders if not exist
os.makedirs(os.path.join(OUTPUT_DIR, "HR"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "LR"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "IR"), exist_ok=True)

# Function to resize and save numpy arrays
def resize_npy_images(input_folder, output_folder, size=(128, 128)):
    files = [f for f in os.listdir(input_folder) if f.endswith('.npy')]
    print(f"Processing {len(files)} files from {input_folder}...")

    for file in files:
        path = os.path.join(input_folder, file)
        arr = np.load(path)

        # Ensure 3D shape
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)

        # Resize using OpenCV
        resized = cv2.resize(arr, size)

        # If image has more than 1 channel, ensure correct shape
        if arr.shape[-1] > 1:
            resized = resized.reshape(size[1], size[0], -1)

        np.save(os.path.join(output_folder, file), resized)

    print(f"✅ Done! Resized files saved in: {output_folder}")

# Process HR, LR, and IR folders
resize_npy_images(os.path.join(BASE_DIR, "HR"), os.path.join(OUTPUT_DIR, "HR"))
resize_npy_images(os.path.join(BASE_DIR, "LR"), os.path.join(OUTPUT_DIR, "LR"))
resize_npy_images(os.path.join(BASE_DIR, "IR"), os.path.join(OUTPUT_DIR, "IR"))

print("🎯 All .npy images resized to 128x128 successfully!")
