import os
import json
import uuid
import numpy as np

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow import keras
from PIL import Image, UnidentifiedImageError

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

MODEL_PATH = "models/brain_tumor_model.keras"
LABEL_PATH = "models/class_names.json"
UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = keras.models.load_model(MODEL_PATH)

with open(LABEL_PATH, "r") as f:
    class_names = json.load(f)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
IMAGE_SIZE = (224, 224)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    probabilities = model.predict(image_array, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))

    results = [
        {
            "class_name": name,
            "probability": float(probability),
            "percentage": f"{probability * 100:.2f}%"
        }
        for name, probability in zip(class_names, probabilities)
    ]

    results.sort(
        key=lambda item: item["probability"],
        reverse=True
    )

    return class_names[predicted_index], results


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    results = None
    image_url = None
    error = None

    if request.method == "POST":
        uploaded_file = request.files.get("mri_image")

        if not uploaded_file or not uploaded_file.filename:
            error = "Please select an MRI image."
        elif not allowed_file(uploaded_file.filename):
            error = "Please upload a JPG, JPEG, or PNG image."
        else:
            filename = secure_filename(uploaded_file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            image_path = os.path.join(
                UPLOAD_FOLDER, unique_filename
            )

            try:
                uploaded_file.save(image_path)

                # Verify that the uploaded file is a readable image.
                with Image.open(image_path) as image:
                    image.verify()

                prediction, results = predict_image(image_path)
                image_url = "/" + image_path.replace("\\", "/")

            except (UnidentifiedImageError, OSError, ValueError):
                if os.path.exists(image_path):
                    os.remove(image_path)

                error = "The uploaded file is not a valid image."

    return render_template(
        "index.html",
        prediction=prediction,
        results=results,
        image_url=image_url,
        error=error
    )


@app.errorhandler(413)
def file_too_large(error):
    return render_template(
        "index.html",
        error="File is too large. Maximum upload size is 10 MB."
    ), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
