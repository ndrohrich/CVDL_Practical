from flask import Flask, send_from_directory, render_template
from flask_socketio import SocketIO
import cv2
import base64
import numpy as np
import time
from PIL import Image
import dill
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sys
from torchvision import transforms

labels=['happy','surprise','sad','angry','disgust','fear']
emo=['😄', '😲', '😢', '😠', '🤢', '😨']
predict_value=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 

def load_model(model_path: str):
    """Loads the emotion classification model from the specified file.

    Args:
        model_path (str): The path to the model file.

    Returns:
        Any: The loaded model.
    """
    if model_path is None:
        if len(sys.argv) > 1:
            model_path = sys.argv[1]
        else:
            raise ValueError("Please provide a model path as an argument.")
    
    print(f"Loading model from {model_path}")
    model=torch.load(model_path)
    model.eval()
    model=model.to('cpu')
    return model

model_path = None
model = load_model(None)

def predict(img: np.ndarray):
    """Predicts emotions from the provided image using the specified model.

    Args:
        model: The emotion classification model to use for prediction.
        img (np.ndarray): The input image as a NumPy array.

    Returns:
        np.ndarray: An array of prediction results.
    """
    
    global predict_value
    global model
    predict_value = model(torch.tensor(img).float())
    predict_value=predict_value#[1]

    
    return labels[np.argmax(predict_value.detach().numpy())]
    







# Flask app setup
app = Flask(__name__,template_folder='demo/static')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', transports=['websocket'])

# Initialize face recognition model
face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Global flag to control flow
received_flag = True  # Initial state allows receiving the first frame


@app.route('/')
def index():
    """Serves the main index.html file.

    Returns:
        Response: The index.html file from the static directory.
    """
    temo=[labels[i]+emo[i]+":"+str(predict_value[i]) for i in range(6)]
    return render_template('index.html',emotion=temo)

# WebSocket event: Receive and process frame
@socketio.on('send_frame')
def handle_frame(data):
    """Handles incoming video frames sent from the client.

    Args:
        data (str): Base64 encoded image data received from the client.

    Returns:
        None
    """
    global received_flag
    try:
        # Ensure flow synchronization, wait for client confirmation
        while not received_flag:
            time.sleep(0.01)  # Wait for client confirmation to prevent high CPU usage

        received_flag = False  # Set flag to False to prevent new frame processing

        # Decode Base64 frame
        frame_data = base64.b64decode(data.split(',')[1])
        np_arr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Convert to grayscale
        gray_frame = frame
        gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_BGR2GRAY)
        # Detect faces and add annotations
        faces = face.detectMultiScale(gray_frame, 1.1, 4)
        modeloutput = "happy"
        # Expand to three channels
        gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        for (x, y, w, h) in faces:
            cv2.rectangle(gray_frame, (x, y), (x + w, y + h), (255, 173, 0), 2)
            # using the cutted face to predict the emotion
            faceimg = np.array(Image.fromarray(gray_frame).crop((x, y, x + w, y + h)).resize((64, 64)))
            faceimg = faceimg.mean(axis=-1)
            faceimg = faceimg.reshape(1, 1, 64, 64)
            
            modeloutput = predict(faceimg)
            cv2.putText(gray_frame, modeloutput, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 173, 0), 5)

        # Encode processed frame
        _, buffer = cv2.imencode('.jpg', gray_frame)
        gray_frame_base64 = base64.b64encode(buffer).decode('utf-8')

        # Send processed image back to client
        socketio.emit('processed_frame', f"data:image/jpeg;base64,{gray_frame_base64}")

    except Exception as e:
        print(f"Error: {e}")


# WebSocket event: Client confirms receipt of frame
@socketio.on('client_received')
def handle_client_received():
    """Handles the event when the client acknowledges receipt of a processed frame.

    Returns:
        None
    """
    global received_flag
    received_flag = True  # Set flag to True to allow server to receive next frame
    # print("client received frame, ready for next frame")


@app.route('/static/<path:path>')
def send_static(path):
    """Serves static files requested by the client.

    Args:
        path (str): The relative path to the static file.

    Returns:
        Response: The requested static file.
    """
    print(path)
    return send_from_directory('src/static', path)


if __name__ == '__main__':
    """Starts the Flask-SocketIO server."""
    print("Starting WebSocket server...")
    socketio.run(app, host='0.0.0.0', port=8080, debug=True)
