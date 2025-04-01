import pickle
import numpy as np
import cv2
import mediapipe as mp
from flask import Flask, Response, render_template

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model_dict = pickle.load(open("model.p", "rb"))
model = model_dict["model"]
max_length = model_dict["max_length"]

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)
labels_dict = {i: chr(65 + i) for i in range(26)}  # A-Z

def process_frame(frame):
    """Extract hand landmarks and make a prediction."""
    data_aux, x_, y_ = [], [], []
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

        # Pad data to match max_length
        data_aux_padded = np.pad(data_aux, (0, max_length - len(data_aux)), mode='constant')
        prediction = model.predict([np.asarray(data_aux_padded)])
        return labels_dict[int(prediction[0])]

    return None

def generate_frames():
    """Capture video frames and make predictions in real-time."""
    cap = cv2.VideoCapture(0)  # Access first available camera
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        predicted_character = process_frame(frame)

        # Draw text on frame
        if predicted_character:
            cv2.putText(frame, predicted_character, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route("/")
def index():
    """Render the homepage."""
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    """Return the camera feed as a video stream."""
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)
