# Brain Tumor Detection Using Deep Learning

An educational brain MRI image classification project using
TensorFlow, MobileNetV2 transfer learning, and Flask.

## Features

- MRI image dataset exploration and preprocessing
- Four-class image classification
- Transfer learning with MobileNetV2
- Training and validation curves
- Test-set evaluation and confusion matrix
- Flask website for image upload and prediction
- Class probability visualization

## Classes

- Glioma
- Meningioma
- No tumor
- Pituitary tumor

The class order is stored in `models/class_names.json`.

## Project Structure

    brain-tumor-detection/
    ├── Brain_Tumor_Detection_From_Scratch.ipynb
    ├── app.py
    ├── requirements.txt
    ├── README.md
    ├── .gitignore
    ├── templates/
    │   └── index.html
    ├── static/
    │   └── uploads/
    └── models/
        ├── brain_tumor_model.keras
        └── class_names.json

The trained model is stored separately and must be downloaded
before running the website if it is not included locally.

## Dataset

Dataset: Brain Tumor MRI Dataset by Masoud Nickparvar.

Source:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Follow the dataset's license and usage conditions.

## Run Locally

1. Install Python and create a virtual environment.
2. Install dependencies:

       pip install -r requirements.txt

3. Download or copy the trained model into `models/`.
4. Ensure `models/class_names.json` is present.
5. Start the website:

       python app.py

6. Open http://127.0.0.1:5000/

## Model Evaluation

Record the actual test accuracy, per-class precision, recall,
F1-score, and confusion matrix after running the notebook.

## Disclaimer

This project is for educational and research purposes only.
It is not a medical diagnostic device and must not be used to
make clinical decisions.
