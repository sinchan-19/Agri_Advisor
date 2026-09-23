from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
from pathlib import Path


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)


# =========================================================
# BASE PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent


# =========================================================
# MODEL PATHS
# =========================================================

# ---------------------------------------------------------
# NEW MAIZE / NON-MAIZE GATE MODEL
# ---------------------------------------------------------

GATE_MODEL_PATH = (
    BASE_DIR
    / "ai_model"
    / "maize_gate_v2.keras"
)


# ---------------------------------------------------------
# MAIZE DISEASE MODEL
# ---------------------------------------------------------

DISEASE_MODEL_PATH = (
    BASE_DIR
    / "ai_model"
    / "maize_disease_model_best.keras"
)


# ---------------------------------------------------------
# DISEASE CLASS FILE
# ---------------------------------------------------------

CLASS_FILE_PATH = (
    BASE_DIR
    / "ai_model"
    / "classes.txt"
)


# =========================================================
# SETTINGS
# =========================================================

# IMPORTANT
#
# New maize gate dataset:
#
# class 0 = maize
# class 1 = not_maize
#
# Therefore:
#
# raw gate output = NOT-MAIZE confidence
# maize confidence = 1 - raw gate output
#
# We start with 50% while testing the new model.
#
# DO NOT use the old 97% threshold here.
# =========================================================

MAIZE_GATE_THRESHOLD = 0.50


# ---------------------------------------------------------
# Image sizes
# ---------------------------------------------------------

DISEASE_IMAGE_SIZE = (224, 224)

GATE_IMAGE_SIZE = (224, 224)


# =========================================================
# LOAD MAIZE GATE MODEL
# =========================================================

print()
print("==============================================")
print("Loading NEW maize / non-maize gate...")
print("==============================================")


if not GATE_MODEL_PATH.exists():

    print("ERROR: Maize gate model not found!")

    print(
        f"Expected location:\n{GATE_MODEL_PATH}"
    )

    maize_gate = None

else:

    try:

        maize_gate = tf.keras.models.load_model(
            GATE_MODEL_PATH
        )

        print(
            "Maize gate loaded successfully."
        )

        print(
            f"Gate model:\n{GATE_MODEL_PATH}"
        )

    except Exception as e:

        print(
            "ERROR: Could not load maize gate."
        )

        print(
            f"Error: {e}"
        )

        maize_gate = None


# =========================================================
# LOAD DISEASE MODEL
# =========================================================

print()
print("==============================================")
print("Loading maize disease model...")
print("==============================================")


if not DISEASE_MODEL_PATH.exists():

    print("ERROR: Disease model not found!")

    print(
        f"Expected location:\n{DISEASE_MODEL_PATH}"
    )

    disease_model = None

else:

    try:

        disease_model = tf.keras.models.load_model(
            DISEASE_MODEL_PATH
        )

        print(
            "Disease model loaded successfully."
        )

        print(
            f"Disease model:\n{DISEASE_MODEL_PATH}"
        )

    except Exception as e:

        print(
            "ERROR: Could not load disease model."
        )

        print(
            f"Error: {e}"
        )

        disease_model = None


# =========================================================
# LOAD DISEASE CLASS NAMES
# =========================================================

classes = []


if CLASS_FILE_PATH.exists():

    with open(
        CLASS_FILE_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        classes = [
            line.strip()
            for line in f
            if line.strip()
        ]


# =========================================================
# FALLBACK CLASSES
# =========================================================

if not classes:

    classes = [
        "common_rust",
        "gray_leaf_spot",
        "healthy",
        "northern_leaf_blight"
    ]


print()
print("==============================================")
print("Disease classes:")
print(classes)
print("==============================================")


# =========================================================
# DISEASE INFORMATION
# =========================================================

DISEASE_INFO = {

    "common_rust": {

        "name": "Common Rust",

        "solution": (
            "Remove heavily infected leaves where practical. "
            "Maintain good field sanitation and avoid excessive "
            "leaf wetness. Use a recommended fungicide when "
            "disease pressure is high."
        ),

        "yield": (
            "Early detection and proper disease management "
            "can help reduce yield loss."
        )
    },


    "gray_leaf_spot": {

        "name": "Gray Leaf Spot",

        "solution": (
            "Improve field sanitation, use resistant varieties "
            "where available, maintain balanced nutrition, "
            "and follow locally recommended fungicide practices."
        ),

        "yield": (
            "Timely management can reduce the risk of significant "
            "yield reduction."
        )
    },


    "healthy": {

        "name": "Healthy",

        "solution": (
            "The maize leaf appears healthy. Continue proper "
            "irrigation, balanced fertilization, field monitoring "
            "and good crop management."
        ),

        "yield": (
            "Healthy plants generally have better potential "
            "for normal yield when other growing conditions "
            "are suitable."
        )
    },


    "northern_leaf_blight": {

        "name": "Northern Leaf Blight",

        "solution": (
            "Remove or manage crop residues where appropriate, "
            "use resistant varieties where available, and apply "
            "recommended fungicide treatment when necessary."
        ),

        "yield": (
            "Early identification and management can help "
            "minimize yield losses."
        )
    }
}


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return send_from_directory(
        BASE_DIR,
        "index.html"
    )


# =========================================================
# STATIC FILES
# =========================================================

@app.route("/<path:filename>")
def static_files(filename):

    file_path = BASE_DIR / filename

    if file_path.exists() and file_path.is_file():

        return send_from_directory(
            BASE_DIR,
            filename
        )

    return jsonify({

        "success": False,

        "message": "File not found."

    }), 404


# =========================================================
# MAIZE / NON-MAIZE GATE
# =========================================================

def check_maize_gate(image_array):

    """
    Checks whether the uploaded image is maize.

    New gate model:

        class 0 = MAIZE
        class 1 = NOT MAIZE

    Therefore:

        raw output = NOT-MAIZE confidence
        maize confidence = 1 - raw output

    IMPORTANT:

    maize_gate_v2.keras already contains
    MobileNetV2 preprocess_input() inside
    the model.

    Therefore this function MUST NOT call
    preprocess_input() again.
    """


    # =====================================================
    # CHECK MODEL
    # =====================================================

    if maize_gate is None:

        return {

            "accepted": False,

            "is_maize": False,

            "maize_confidence": 0.0,

            "non_maize_confidence": 100.0,

            "message": (
                "Maize detection model is not available."
            )
        }


    # =====================================================
    # CLEAN IMAGE
    # =====================================================

    image_array = np.clip(
        image_array,
        0,
        255
    ).astype(np.float32)


    # =====================================================
    # RESIZE IMAGE
    # =====================================================

    gate_image = tf.image.resize(
        image_array,
        GATE_IMAGE_SIZE
    )


    # =====================================================
    # IMPORTANT:
    #
    # DO NOT CALL:
    #
    # tf.keras.applications.mobilenet_v2.preprocess_input()
    #
    # The new model already performs preprocessing.
    # =====================================================

    gate_image = tf.cast(
        gate_image,
        tf.float32
    )


    # =====================================================
    # ADD BATCH DIMENSION
    # =====================================================

    gate_image = tf.expand_dims(
        gate_image,
        axis=0
    )


    # =====================================================
    # PREDICT
    # =====================================================

    gate_prediction = maize_gate.predict(
        gate_image,
        verbose=0
    )


    # =====================================================
    # CONVERT OUTPUT TO FLOAT
    # =====================================================

    gate_value = float(
        np.squeeze(gate_prediction)
    )


    # =====================================================
    # CLASS INTERPRETATION
    # =====================================================

    # class 0 = maize
    # class 1 = not_maize

    non_maize_confidence = gate_value

    maize_confidence = (
        1.0 - non_maize_confidence
    )


    # =====================================================
    # SAFETY CLAMP
    # =====================================================

    maize_confidence = float(
        np.clip(
            maize_confidence,
            0.0,
            1.0
        )
    )


    non_maize_confidence = float(
        np.clip(
            non_maize_confidence,
            0.0,
            1.0
        )
    )


    # =====================================================
    # PERCENTAGES
    # =====================================================

    maize_percent = (
        maize_confidence * 100
    )

    non_maize_percent = (
        non_maize_confidence * 100
    )


    # =====================================================
    # TERMINAL INFORMATION
    # =====================================================

    print()
    print("==============================================")
    print("MAIZE GATE RESULT")
    print("==============================================")


    print(
        f"Raw gate output     : "
        f"{gate_value:.6f}"
    )


    print(
        f"Maize Confidence    : "
        f"{maize_percent:.2f}%"
    )


    print(
        f"Non-Maize Confidence: "
        f"{non_maize_percent:.2f}%"
    )


    print(
        f"Required Confidence : "
        f"{MAIZE_GATE_THRESHOLD * 100:.2f}%"
    )


    # =====================================================
    # HARD REJECTION
    # =====================================================

    if maize_confidence < MAIZE_GATE_THRESHOLD:

        print()
        print(
            "REJECTED - NOT A MAIZE IMAGE"
        )

        print(
            "Disease model will NOT be executed."
        )

        print(
            "=============================================="
        )


        return {

            "accepted": False,

            "is_maize": False,

            "maize_confidence": maize_percent,

            "non_maize_confidence": (
                non_maize_percent
            ),

            "message": (
                "This is not a correct maize crop image. "
                "Please upload a clear maize leaf/crop image."
            )
        }


    # =====================================================
    # ACCEPT MAIZE
    # =====================================================

    print()
    print(
        "ACCEPTED - MAIZE IMAGE"
    )

    print(
        "Disease model can now be executed."
    )

    print(
        "=============================================="
    )


    return {

        "accepted": True,

        "is_maize": True,

        "maize_confidence": maize_percent,

        "non_maize_confidence": (
            non_maize_percent
        ),

        "message": "Valid maize image."
    }


# =========================================================
# LOAD UPLOADED IMAGE
# =========================================================

def load_uploaded_image(file):

    """
    Converts uploaded image into a NumPy RGB array.
    """

    try:

        # -------------------------------------------------
        # READ IMAGE BYTES
        # -------------------------------------------------

        image_bytes = file.read()


        if not image_bytes:

            return None


        # -------------------------------------------------
        # DECODE IMAGE
        # -------------------------------------------------

        image_tensor = tf.io.decode_image(
            image_bytes,
            channels=3,
            expand_animations=False
        )


        # -------------------------------------------------
        # CONVERT TO NUMPY
        # -------------------------------------------------

        image_array = image_tensor.numpy()


        # -------------------------------------------------
        # ENSURE UINT8
        # -------------------------------------------------

        image_array = np.clip(
            image_array,
            0,
            255
        ).astype(np.uint8)


        return image_array


    except Exception as e:

        print(
            f"Image loading error: {e}"
        )

        return None


# =========================================================
# DISEASE PREDICTION
# =========================================================

def predict_disease(image_array):

    """
    Runs the disease model.

    IMPORTANT:

    This function is called ONLY after
    the maize gate accepts the image.
    """


    # =====================================================
    # CHECK MODEL
    # =====================================================

    if disease_model is None:

        return {

            "success": False,

            "message": (
                "Disease model is not available."
            )
        }


    # =====================================================
    # RESIZE
    # =====================================================

    image = tf.image.resize(
        image_array,
        DISEASE_IMAGE_SIZE
    )


    # =====================================================
    # CONVERT FLOAT
    # =====================================================

    image = tf.cast(
        image,
        tf.float32
    )


    # =====================================================
    # ADD BATCH DIMENSION
    # =====================================================

    image = tf.expand_dims(
        image,
        axis=0
    )


    # =====================================================
    # RUN DISEASE MODEL
    # =====================================================

    prediction = disease_model.predict(
        image,
        verbose=0
    )


    # =====================================================
    # CONVERT PREDICTION
    # =====================================================

    prediction = np.asarray(
        prediction
    )

    prediction = np.squeeze(
        prediction
    )


    # =====================================================
    # HANDLE BINARY MODEL
    # =====================================================

    if prediction.ndim == 0:

        value = float(
            prediction
        )


        if len(classes) == 2:

            if value >= 0.5:

                class_index = 1

                confidence = value

            else:

                class_index = 0

                confidence = (
                    1.0 - value
                )

        else:

            class_index = 0

            confidence = value


    # =====================================================
    # HANDLE MULTI-CLASS MODEL
    # =====================================================

    else:

        prediction = prediction.astype(
            np.float64
        )


        prediction_sum = np.sum(
            prediction
        )


        # -------------------------------------------------
        # CHECK WHETHER VALUES ARE PROBABILITIES
        # -------------------------------------------------

        if (
            np.any(prediction < 0)
            or
            np.any(prediction > 1)
            or
            not np.isclose(
                prediction_sum,
                1.0,
                atol=0.01
            )
        ):

            exp_prediction = np.exp(
                prediction
                -
                np.max(prediction)
            )


            prediction = (
                exp_prediction
                /
                np.sum(exp_prediction)
            )


        # -------------------------------------------------
        # HIGHEST PROBABILITY
        # -------------------------------------------------

        class_index = int(
            np.argmax(prediction)
        )


        confidence = float(
            prediction[class_index]
        )


    # =====================================================
    # CHECK CLASS INDEX
    # =====================================================

    if class_index >= len(classes):

        class_index = 0


    # =====================================================
    # CLASS NAME
    # =====================================================

    class_name = classes[
        class_index
    ]


    # =====================================================
    # DISEASE DETAILS
    # =====================================================

    disease_details = DISEASE_INFO.get(

        class_name,

        {

            "name": class_name,

            "solution": (
                "Please consult a local agricultural "
                "expert for appropriate management."
            ),

            "yield": (
                "Yield impact depends on disease severity "
                "and crop conditions."
            )
        }
    )


    # =====================================================
    # CONFIDENCE
    # =====================================================

    confidence_percent = (
        confidence * 100
    )


    # =====================================================
    # RETURN DISEASE RESULT
    # =====================================================

    return {

        "success": True,

        "disease": class_name,

        "disease_name": (
            disease_details["name"]
        ),

        "confidence": round(
            confidence_percent,
            2
        ),

        "solution": (
            disease_details["solution"]
        ),

        "yield": (
            disease_details["yield"]
        )
    }


# =========================================================
# PREDICT API
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    print()
    print("==============================================")
    print("NEW IMAGE RECEIVED")
    print("==============================================")


    # =====================================================
    # CHECK FILE
    # =====================================================

    if "file" not in request.files:

        return jsonify({

            "success": False,

            "is_maize": False,

            "message": (
                "No image file was uploaded."
            )

        }), 400


    file = request.files[
        "file"
    ]


    # =====================================================
    # CHECK FILE NAME
    # =====================================================

    if not file.filename:

        return jsonify({

            "success": False,

            "is_maize": False,

            "message": (
                "Please select an image."
            )

        }), 400


    # =====================================================
    # LOAD IMAGE
    # =====================================================

    image_array = load_uploaded_image(
        file
    )


    if image_array is None:

        return jsonify({

            "success": False,

            "is_maize": False,

            "message": (
                "Unable to read this image. "
                "Please upload a valid JPG, JPEG or PNG image."
            )

        }), 400


    print(
        f"Image shape: {image_array.shape}"
    )


    # =====================================================
    # STEP 1
    #
    # MAIZE / NON-MAIZE CHECK
    # =====================================================

    print()

    print(
        "STEP 1: "
        "CHECKING WHETHER IMAGE IS MAIZE..."
    )


    gate_result = check_maize_gate(
        image_array
    )


    # =====================================================
    # HARD STOP
    #
    # VERY IMPORTANT:
    #
    # If image is NOT maize:
    #
    # RETURN IMMEDIATELY
    #
    # Disease model is NOT called.
    # =====================================================

    if not gate_result["accepted"]:

        print()

        print(
            "REQUEST STOPPED"
        )

        print(
            "Reason: "
            "Image failed maize verification."
        )

        print(
            "Disease model was NOT called."
        )

        print(
            "=============================================="
        )


        return jsonify({

            "success": False,

            "is_maize": False,

            "maize_confidence": round(

                gate_result[
                    "maize_confidence"
                ],

                2
            ),

            "message": (
                gate_result[
                    "message"
                ]
            )

        }), 200


    # =====================================================
    # STEP 2
    #
    # ONLY NOW RUN DISEASE MODEL
    # =====================================================

    print()

    print(
        "STEP 2: MAIZE VERIFIED"
    )

    print(
        "Running disease model..."
    )


    disease_result = predict_disease(
        image_array
    )


    # =====================================================
    # CHECK DISEASE RESULT
    # =====================================================

    if not disease_result["success"]:

        return jsonify({

            "success": False,

            "is_maize": True,

            "maize_confidence": round(

                gate_result[
                    "maize_confidence"
                ],

                2
            ),

            "message": (
                disease_result[
                    "message"
                ]
            )

        }), 500


    # =====================================================
    # FINAL RESULT
    # =====================================================

    print()

    print(
        "=============================================="
    )

    print(
        "FINAL RESULT"
    )

    print(
        "=============================================="
    )


    print(

        f"Maize Confidence: "
        f"{gate_result['maize_confidence']:.2f}%"

    )


    print(

        f"Disease: "
        f"{disease_result['disease_name']}"

    )


    print(

        f"Disease Confidence: "
        f"{disease_result['confidence']:.2f}%"

    )


    print(
        "=============================================="
    )


    # =====================================================
    # RETURN FINAL JSON
    # =====================================================

    return jsonify({

        "success": True,

        "is_maize": True,

        "maize_confidence": round(

            gate_result[
                "maize_confidence"
            ],

            2
        ),

        "disease": (
            disease_result[
                "disease"
            ]
        ),

        "disease_name": (
            disease_result[
                "disease_name"
            ]
        ),

        "confidence": (
            disease_result[
                "confidence"
            ]
        ),

        "solution": (
            disease_result[
                "solution"
            ]
        ),

        "yield": (
            disease_result[
                "yield"
            ]
        ),

        "message": (
            "Maize image verified successfully."
        )
    })


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status": "running",

        "maize_gate_loaded": (
            maize_gate is not None
        ),

        "disease_model_loaded": (
            disease_model is not None
        ),

        "classes": classes,

        "maize_threshold": (
            MAIZE_GATE_THRESHOLD * 100
        ),

        "gate_model": (
            GATE_MODEL_PATH.name
        )
    })


# =========================================================
# RUN FLASK
# =========================================================

if __name__ == "__main__":

    print()

    print(
        "=============================================="
    )

    print(
        "AGRI ADVISOR"
    )

    print(
        "MAIZE DISEASE DETECTION"
    )

    print(
        "=============================================="
    )


    print(

        f"Maize gate threshold: "
        f"{MAIZE_GATE_THRESHOLD * 100:.0f}%"

    )


    print(

        f"Gate model: "
        f"{GATE_MODEL_PATH}"

    )


    print(

        f"Disease model: "
        f"{DISEASE_MODEL_PATH}"

    )


    print()

    print(
        "Server starting..."
    )


    print(
        "Open: http://127.0.0.1:5000"
    )


    print(
        "=============================================="
    )


    app.run(

        host="127.0.0.1",

        port=5000,

        debug=False

    )