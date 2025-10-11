import os
import cv2
import numpy as np

# ------------------------------
# CONFIGURATION
# ------------------------------
hr_folder = r"C:\Users\ayush\Downloads\alignimages\HR"
lr_folder = r"C:\Users\ayush\Downloads\alignimages\LR"
ir_folder = r"C:\Users\ayush\Downloads\alignimages\IR"

output_base = r"C:\Users\ayush\Downloads\alignimages\preprocessed_ordered"
size = (256, 256)  # Resize to this (width, height)

# Create output folders
hr_out = os.path.join(output_base, "HR")
lr_out = os.path.join(output_base, "LR")
ir_out = os.path.join(output_base, "IR")
for folder in [hr_out, lr_out, ir_out]:
    os.makedirs(folder, exist_ok=True)

# ------------------------------
# HELPER FUNCTION
# ------------------------------
def preprocess_image(img, resize_dim, normalize=True, keep_rgb=True):
    img_resized = cv2.resize(img, resize_dim)

    if not keep_rgb and len(img_resized.shape) == 3:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    if normalize:
        img_resized = img_resized.astype(np.float32) / 255.0

    return img_resized

# ------------------------------
# GET COMMON FILENAMES
# ------------------------------
hr_files = {os.path.splitext(f)[0]: f for f in os.listdir(hr_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
lr_files = {os.path.splitext(f)[0]: f for f in os.listdir(lr_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
ir_files = {os.path.splitext(f)[0]: f for f in os.listdir(ir_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}

# Get only files present in all three
common_names = sorted(list(set(hr_files.keys()) & set(lr_files.keys()) & set(ir_files.keys())))
print(f"✅ Found {len(common_names)} common image sets")

# ------------------------------
# PROCESS AND SAVE IN SAME ORDER
# ------------------------------
for idx, name in enumerate(common_names, start=1):
    new_name = f"{idx:04d}.npy"

    # HR
    hr_img = cv2.imread(os.path.join(hr_folder, hr_files[name]))
    hr_pre = preprocess_image(hr_img, size, normalize=True, keep_rgb=True)
    np.save(os.path.join(hr_out, new_name), hr_pre)

    # LR
    lr_img = cv2.imread(os.path.join(lr_folder, lr_files[name]))
    lr_pre = preprocess_image(lr_img, size, normalize=True, keep_rgb=True)
    np.save(os.path.join(lr_out, new_name), lr_pre)

    # IR
    ir_img = cv2.imread(os.path.join(ir_folder, ir_files[name]), cv2.IMREAD_GRAYSCALE)
    ir_pre = preprocess_image(ir_img, size, normalize=True, keep_rgb=False)
    np.save(os.path.join(ir_out, new_name), ir_pre)

print("\n🎯 Preprocessing complete! Files saved in same order.")
print(f"Output Folder: {output_base}")
print(f" ├── HR/ ({len(os.listdir(hr_out))} files)")
print(f" ├── LR/ ({len(os.listdir(lr_out))} files)")
print(f" └── IR/ ({len(os.listdir(ir_out))} files)")
