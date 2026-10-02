
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps, UnidentifiedImageError
import numpy as np
import io
import os

app = Flask(__name__)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "deep_learning_model.h5"

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB


# --------------------------------------------------
# Load trained CNN model
# --------------------------------------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}. "
        "Place deep_learning_model.h5 beside flask_api.py."
    )

model = load_model(MODEL_PATH, compile=False)

print("CNN model loaded successfully.")


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

def preprocess_image(image):
    """
    Convert an uploaded image into a model input.

    Expected model input:
        (1, 28, 28, 1)

    Expected pixel range:
        [0, 1]

    Assumes the CNN was trained on grayscale MNIST
    images with white digits on a black background.
    """

    image = image.convert("L")

    # Preserve native 28 x 28 MNIST images
    if image.size == (28, 28):
        arr = np.asarray(image, dtype=np.float32)

        border = np.concatenate([
            arr[0, :],
            arr[-1, :],
            arr[:, 0],
            arr[:, -1]
        ])

        # MNIST normally has a dark background
        if border.mean() > 127:
            arr = 255.0 - arr

        arr = arr / 255.0

        return arr.reshape(1, 28, 28, 1)

    # Handle larger uploaded images
    arr = np.asarray(image, dtype=np.float32)

    border = np.concatenate([
        arr[0, :],
        arr[-1, :],
        arr[:, 0],
        arr[:, -1]
    ])

    # Convert light-background images to dark-background
    # images only when the border indicates a light background.
    if border.mean() > 127:
        image = ImageOps.invert(image)

    # Fit the digit into a 20 x 20 area while preserving
    # its aspect ratio, then center it on a 28 x 28 canvas.
    image.thumbnail((20, 20), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), color=0)

    x_offset = (28 - image.width) // 2
    y_offset = (28 - image.height) // 2

    canvas.paste(image, (x_offset, y_offset))

    arr = np.asarray(canvas, dtype=np.float32) / 255.0

    return arr.reshape(1, 28, 28, 1)


# --------------------------------------------------
# Home endpoint
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "success": True,
        "message": "MNIST CNN Flask API is running.",
        "endpoints": {
            "health": "/health",
            "predict": "/predict"
        }
    }), 200


# --------------------------------------------------
# Health-check endpoint
# --------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "success": True,
        "status": "healthy",
        "model_loaded": model is not None
    }), 200


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    # Check whether an image was supplied
    if "file" not in request.files:
        return jsonify({
            "success": False,
            "error": "Missing image. Upload it using the 'file' field."
        }), 400

    uploaded_file = request.files["file"]

    if not uploaded_file.filename:
        return jsonify({
            "success": False,
            "error": "No file selected."
        }), 400

    try:
        # Read the uploaded image safely
        image_bytes = uploaded_file.read()

        if not image_bytes:
            return jsonify({
                "success": False,
                "error": "The uploaded file is empty."
            }), 400

        image = Image.open(io.BytesIO(image_bytes))
        image.load()

        # Preprocess image
        processed_image = preprocess_image(image)

        # Run inference
        predictions = model.predict(
            processed_image,
            verbose=0
        )

        probabilities = np.asarray(predictions[0])

        # Validate model output
        if probabilities.shape != (10,):
            return jsonify({
                "success": False,
                "error": (
                    "Unexpected model output shape. "
                    "Expected probabilities for digits 0 to 9."
                )
            }), 500

        if not np.all(np.isfinite(probabilities)):
            return jsonify({
                "success": False,
                "error": "The model returned invalid prediction values."
            }), 500

        # The model is assumed to output 10-class softmax scores
        predicted_digit = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_digit]) * 100

        class_probabilities = {
            str(i): round(float(probabilities[i]), 6)
            for i in range(10)
        }

        return jsonify({
            "success": True,
            "predicted_digit": predicted_digit,
            "confidence": round(confidence, 2),
            "probabilities": class_probabilities
        }), 200

    except (UnidentifiedImageError, OSError, ValueError) as error:
        return jsonify({
            "success": False,
            "error": f"Unable to process the uploaded image: {error}"
        }), 400

    except Exception:
        app.logger.exception("Prediction failed")

        return jsonify({
            "success": False,
            "error": "Prediction failed. Check the Flask server logs."
        }), 500


# --------------------------------------------------
# Handle oversized uploads
# --------------------------------------------------

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "success": False,
        "error": "File is too large. Maximum allowed size is 10 MB."
    }), 413


# --------------------------------------------------
# Run Flask server
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
