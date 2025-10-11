import os
import shutil

# -------------------------------
#  ALIGN CROP IMAGES BY FILENAME
# -------------------------------

# 🗂️ Input folder paths (change according to your setup)
infrared_path = r"C:\Users\ayush\Downloads\RoadScene-master\RoadScene-master\cropinfrared"
lr_visible_path = r"C:\Users\ayush\Downloads\RoadScene-master\RoadScene-master\crop_LR_visible"
hr_visible_path = r"C:\Users\ayush\Downloads\RoadScene-master\RoadScene-master\crop_HR_visible"

# 🗂️ Output base folder
output_base = r"C:\Users\ayush\Downloads\align"

# Create output directories
os.makedirs(output_base, exist_ok=True)
out_ir = os.path.join(output_base, "IR")
out_lr = os.path.join(output_base, "LR")
out_hr = os.path.join(output_base, "HR")
for path in [out_ir, out_lr, out_hr]:
    os.makedirs(path, exist_ok=True)

print("🔍 Reading files from folders...")

# Read filenames without extensions
ir_files = {os.path.splitext(f)[0]: f for f in os.listdir(infrared_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
lr_files = {os.path.splitext(f)[0]: f for f in os.listdir(lr_visible_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
hr_files = {os.path.splitext(f)[0]: f for f in os.listdir(hr_visible_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}

# Find common filenames present in all three folders
common_names = sorted(list(set(ir_files.keys()) & set(lr_files.keys()) & set(hr_files.keys())))

print(f"✅ Found {len(common_names)} matched image sets")

# Copy matched files and rename sequentially
for idx, name in enumerate(common_names, start=1):
    ir_file = ir_files[name]
    lr_file = lr_files[name]
    hr_file = hr_files[name]

    new_name = f"{idx:04d}.png"

    shutil.copy(os.path.join(infrared_path, ir_file), os.path.join(out_ir, new_name))
    shutil.copy(os.path.join(lr_visible_path, lr_file), os.path.join(out_lr, new_name))
    shutil.copy(os.path.join(hr_visible_path, hr_file), os.path.join(out_hr, new_name))

print("\n🎯 Alignment complete!")
print(f"All aligned images saved under: {output_base}")
print("Structure:")
print(f" ├── IR/   ({len(os.listdir(out_ir))} files)")
print(f" ├── LR/   ({len(os.listdir(out_lr))} files)")
print(f" └── HR/   ({len(os.listdir(out_hr))} files)")
