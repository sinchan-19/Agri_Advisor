import sys
import tensorflow as tf
import numpy as np
from pathlib import Path

# -----------------------------------------
# Settings
# -----------------------------------------

MODEL_PATH = "ai_model/maize_disease_model.keras"
CLASSES_PATH = "ai_model/classes.txt"

IMAGE_SIZE = (224, 224)


# -----------------------------------------
# Check image argument
# -----------------------------------------

if len(sys.argv) < 2:
    print("Please provide an image path.")
    print()
    print("Example:")
    print(
        'python predict.py "dataset/maize_dataset/healthy/image.jpg"'
    )
    sys.exit(1)


image_path = Path(sys.argv[1])


if not image_path.exists():
    print("Image not found:")
    print(image_path)
    sys.exit(1)


# -----------------------------------------
# Load model
# -----------------------------------------

print("Loading AI model...")

model = tf.keras.models.load_model(MODEL_PATH)


# -----------------------------------------
# Load class names
# -----------------------------------------

with open(CLASSES_PATH, "r") as file:
    class_names = [
        line.strip()
        for line in file
        if line.strip()
    ]


# -----------------------------------------
# Load image
# -----------------------------------------

print("Loading image...")

image = tf.keras.utils.load_img(
    image_path,
    target_size=IMAGE_SIZE
)


image_array = tf.keras.utils.img_to_array(image)

image_array = np.expand_dims(
    image_array,
    axis=0
)


# -----------------------------------------
# Prediction
# -----------------------------------------

predictions = model.predict(
    image_array,
    verbose=0
)

predicted_index = np.argmax(predictions[0])

predicted_class = class_names[predicted_index]

confidence = float(
    predictions[0][predicted_index]
) * 100


# -----------------------------------------
# Result
# -----------------------------------------

print()
print("================================")
print("       MAIZE AI RESULT")
print("================================")

print("Prediction :", predicted_class)
print(f"Confidence : {confidence:.2f}%")

print("================================")