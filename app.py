from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import pickle

# Load trained model
model = load_model("model.h5")

# Load saved label encoder
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/prediction.html')
def prediction():
    return render_template("prediction.html")

@app.route('/result', methods=['POST'])
def result():
    if 'file' not in request.files:
        return "No file uploaded"
    
    file = request.files['file']
    if file.filename == '':
        return "No file selected"
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Read and preprocess image
    img = cv2.imread(filepath)
    img = cv2.resize(img, (128, 128))       # must match model training size
    img = img / 255.0                        # normalize
    img = np.expand_dims(img, axis=0)       # add batch dimension

    # Predict
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    class_name = encoder.inverse_transform([predicted_class])[0]

    return render_template("logout.html", prediction=class_name)

if __name__ == "__main__":
    app.run(debug=True)
