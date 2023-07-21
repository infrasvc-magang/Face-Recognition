# app.py
from flask import *
import face_recognition
import cv2
import json
import numpy as np
import math
from keras.models import load_model
from recognition import run_recognition

app = Flask(__name__)

# Load age and emotion models
# ...

# Rest of your face recognition code
# ...

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognition', methods=['POST'])
def face_recognition():
    if request.method == 'POST':
        # Get the image data from the frontend
        image_data = request.form['image_data']

        # Convert the base64 encoded image data to numpy array
        nparr = np.fromstring(image_data.decode('base64'), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Perform face recognition and return the result
        # Modify the FaceRecognition class to accept the frame data
        # and return the face recognition result
        result = perform_face_recognition(frame)

        # Prepare the response
        response = {
            'result': result
        }

        # Convert the response to JSON and return it
        return jsonify(response)

def perform_face_recognition(frame):
    # Your face recognition logic
    # ...

    # Return the recognized names and other details as needed
    return recognized_names, emo_prediction, age_prediction

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
