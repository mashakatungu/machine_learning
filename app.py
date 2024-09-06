from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Load the trained model
model = load_model(r'C:\pythonProject\potato diseases detection\potatoes_model_trained.h5')


# Function to load and resize the uploaded image
def load_and_resize_image(file_path, target_shape=(128, 128)):
    image = cv2.imread(file_path)
    resized_image = cv2.resize(image, target_shape)
    return resized_image


# Function to predict disease and provide solutions
def predict_and_provide_solution(image_path, threshold=0.5):
    image = load_and_resize_image(image_path)
    image = image / 255.0  # Normalize the image
    image_reshaped = np.expand_dims(image, axis=0)

    # Predict the disease class
    predictions = model.predict(image_reshaped)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    if confidence < threshold:
        disease_info = {'name': 'Unknown', 'solution': 'No solution available for unknown image.'}
    else:
        disease_info = provide_disease_info(predicted_class)

    return disease_info


# Provide disease information and solutions
def provide_disease_info(predicted_class):
    disease_info = {
        0: {'name': 'Early Blight', 'solution': 'Apply fungicide and crop rotation.'},
        1: {'name': 'Late Blight', 'solution': 'Use resistant varieties and proper irrigation.'},
        2: {'name': 'Healthy', 'solution': 'The plant is healthy!'}
    }
    return disease_info.get(predicted_class, {'name': 'Unknown', 'solution': 'No solution available.'})


# Route for uploading the image and predicting the disease
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the file is uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            file_path = os.path.join('static/uploads', file.filename)
            file.save(file_path)
            result = predict_and_provide_solution(file_path)
            return render_template('result.html', image_url=file_path, result=result)

    return render_template('index.html')

@app.route('/result')
def result():
    # Pass image_url (relative path) and prediction result to the template
    return render_template('result.html', image_url='uploads/your_image.png', result=result_data)


if __name__ == '__main__':
    app.run(debug=True,port = 7070)
