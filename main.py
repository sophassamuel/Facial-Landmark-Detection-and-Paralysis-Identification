from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import base64
from tensorflow import keras
import re

app = Flask(__name__)

# Load the trained model
model = keras.models.load_model('Classification-final.h5')

# Function to preprocess input images
def preprocess_image(image):
    # Resize image to match model input shape
    img_width, img_height = 150, 150  # Update with the expected input shape of the model
    image = cv2.resize(image, (img_width, img_height))
    # Normalize pixel values
    image = image.astype('float32') / 255.0
    # Expand dimensions to match batch size
    image = np.expand_dims(image, axis=0)
    return image

# Function to perform classification
def classify_image(image):
    # Preprocess input image
    processed_image = preprocess_image(image)
    # Predict class probabilities
    probabilities = model.predict(processed_image)
    # Get class label (assuming binary classification)
    predicted_class = "Paralysed" if probabilities[0] >= 0.5 else "Normal"
    return predicted_class

# Function to decode base64-encoded image data with correct padding
def decode_base64_with_padding(image_data):
    # Add padding if missing
    missing_padding = len(image_data) % 4
    if missing_padding != 0:
        image_data += '=' * (4 - missing_padding)
    # Decode base64
    return base64.b64decode(image_data)

@app.route('/', methods=['POST', 'GET'])
def classify():
    if request.method == 'POST':
        # Read image data from the request
        data = request.get_json()
        image_data = data.get('image_data')

        # Ensure correct length of base64-encoded string
        image_data = re.sub('^data:image/.+;base64,', '', image_data)
        img_bytes = decode_base64_with_padding(image_data)

        # Convert bytes to NumPy array
        nparr = np.frombuffer(img_bytes, np.uint8)

        # Decode the image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Perform classification on the image
        predicted_class = classify_image(img)

        # Return classification result as JSON
        return jsonify({'prediction': predicted_class})
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
