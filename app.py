from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2
import os

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model("face_emotion_model.h5")

# Emotion labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Home route
@app.route('/')
def home():
    return '''
        <h1>Face Emotion Detection</h1>
        <p>Upload an image to predict the emotion.</p>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*">
            <input type="submit" value="Upload">
        </form>
    '''

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Load image and prepare for model
    img = image.load_img(file_path, target_size=(48, 48), color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Predict emotion
    prediction = model.predict(img_array)
    emotion_label = emotions[np.argmax(prediction)]

    return f"<h2>Predicted Emotion: {emotion_label}</h2>"

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
