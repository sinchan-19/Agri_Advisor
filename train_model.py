import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

# ============================================================
# 1. SETTINGS
# ============================================================

DATASET_DIR = Path("dataset/maize_dataset")
MODEL_DIR = Path("ai_model")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 123


# ============================================================
# 2. CHECK DATASET
# ============================================================

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_DIR}"
    )

print("\nDataset found:")
print(DATASET_DIR.resolve())


# ============================================================
# 3. LOAD DATASET
# ============================================================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.20,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)


# ============================================================
# 4. GET CLASS NAMES
# ============================================================

class_names = train_dataset.class_names

print("\nClasses detected by AI:")
for number, name in enumerate(class_names):
    print(number, "=", name)


# ============================================================
# 5. IMPROVE DATA LOADING SPEED
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ============================================================
# 6. DATA AUGMENTATION
# ============================================================

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="data_augmentation")


# ============================================================
# 7. CREATE AI MODEL
# ============================================================

model = keras.Sequential([

    layers.Input(shape=(224, 224, 3)),

    data_augmentation,

    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(256, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(
        len(class_names),
        activation="softmax"
    )
])


# ============================================================
# 8. COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 9. SHOW MODEL
# ============================================================

model.summary()


# ============================================================
# 10. CREATE MODEL DIRECTORY
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 11. TRAIN MODEL
# ============================================================

print("\n====================================")
print("Starting maize disease training...")
print("====================================\n")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)


# ============================================================
# 12. SAVE MODEL
# ============================================================

model_path = MODEL_DIR / "maize_disease_model.keras"

model.save(model_path)


# ============================================================
# 13. SAVE CLASS NAMES
# ============================================================

classes_file = MODEL_DIR / "classes.txt"

with open(classes_file, "w") as file:

    for class_name in class_names:
        file.write(class_name + "\n")


# ============================================================
# 14. FINAL RESULT
# ============================================================

print("\n====================================")
print("TRAINING COMPLETE!")
print("====================================")

print("\nModel saved at:")
print(model_path.resolve())

print("\nClasses saved at:")
print(classes_file.resolve())

print("\nYour AI can detect:")

for class_name in class_names:
    print("-", class_name)