import tensorflow as tf
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


# ==============================
# 1. Load model
# ==============================

model_path = Path("ai_model/maize_disease_model.keras")

model = tf.keras.models.load_model(model_path)

print("Model loaded successfully!")


# ==============================
# 2. Load test dataset
# ==============================

test_dir = Path("dataset/disease_data/test")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

class_names = test_dataset.class_names

print("\nClasses:")
for i, name in enumerate(class_names):
    print(i, "=", name)


# ==============================
# 3. Get actual labels
# ==============================

y_true = np.concatenate([
    y.numpy() for x, y in test_dataset
])


# ==============================
# 4. Get predictions
# ==============================

predictions = model.predict(test_dataset)

y_pred = np.argmax(predictions, axis=1)


# ==============================
# 5. Classification report
# ==============================

print("\n========== CLASSIFICATION REPORT ==========\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
)


# ==============================
# 6. Confusion matrix
# ==============================

cm = confusion_matrix(y_true, y_pred)

print("\n========== CONFUSION MATRIX ==========\n")
print(cm)


# ==============================
# 7. Display confusion matrix
# ==============================

plt.figure(figsize=(8, 6))

plt.imshow(cm)

plt.title("Maize Disease Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=45
)

plt.yticks(
    range(len(class_names)),
    class_names
)

for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig("confusion_matrix.png")

plt.show()

print("\nConfusion matrix saved as: confusion_matrix.png")
