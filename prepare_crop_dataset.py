import os
import shutil

SOURCE = "dataset/maize_dataset"
DESTINATION = "dataset/crop_detection/maize"

classes = [
    "common_rust",
    "gray_leaf_spot",
    "healthy",
    "northern_leaf_blight"
]

valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

os.makedirs(DESTINATION, exist_ok=True)

total = 0

for class_name in classes:
    source_folder = os.path.join(SOURCE, class_name)

    if not os.path.exists(source_folder):
        print(f"Folder not found: {source_folder}")
        continue

    class_count = 0

    for filename in os.listdir(source_folder):
        if filename.lower().endswith(valid_extensions):

            source_file = os.path.join(source_folder, filename)

            # Add class name to filename to avoid duplicate names
            new_filename = f"{class_name}_{filename}"
            destination_file = os.path.join(DESTINATION, new_filename)

            shutil.copy2(source_file, destination_file)

            class_count += 1
            total += 1

    print(f"{class_name}: {class_count} images copied")

print("--------------------------------")
print(f"Total maize images copied: {total}")
print("Original dataset was NOT changed.")