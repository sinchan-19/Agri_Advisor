import os
import shutil
import random

SOURCE = "dataset/maize_dataset"
DESTINATION = "dataset/disease_data"

CLASSES = [
    "common_rust",
    "gray_leaf_spot",
    "healthy",
    "northern_leaf_blight"
]

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

random.seed(42)

for class_name in CLASSES:

    source_folder = os.path.join(SOURCE, class_name)

    train_folder = os.path.join(DESTINATION, "train", class_name)
    validation_folder = os.path.join(DESTINATION, "validation", class_name)
    test_folder = os.path.join(DESTINATION, "test", class_name)

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(validation_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    images = [
        filename
        for filename in os.listdir(source_folder)
        if filename.lower().endswith(VALID_EXTENSIONS)
    ]

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    validation_end = train_end + int(total * VALIDATION_RATIO)

    train_images = images[:train_end]
    validation_images = images[train_end:validation_end]
    test_images = images[validation_end:]

    for filename in train_images:
        shutil.copy2(
            os.path.join(source_folder, filename),
            os.path.join(train_folder, filename)
        )

    for filename in validation_images:
        shutil.copy2(
            os.path.join(source_folder, filename),
            os.path.join(validation_folder, filename)
        )

    for filename in test_images:
        shutil.copy2(
            os.path.join(source_folder, filename),
            os.path.join(test_folder, filename)
        )

    print(f"\n{class_name}")
    print(f"Total: {total}")
    print(f"Training: {len(train_images)}")
    print(f"Validation: {len(validation_images)}")
    print(f"Testing: {len(test_images)}")

print("\n--------------------------------")
print("Dataset splitting completed.")
print("Original maize_dataset was NOT changed.")