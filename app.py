from flask import Flask, Response, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from flask_cors import CORS   # ✅ NEW

app = Flask(__name__)
CORS(app)  # ✅ อนุญาตให้ FE ต่าง origin เรียก API ได้

# --- Load model ---
model = load_model("hand_sign_model_final.h5")
with open("label_binarizer_final.pkl", "rb") as f:
    lb = pickle.load(f)

# --- Mediapipe ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)

# --- Variables ---
current_text = ""
last_pred = None
stable_count = 0
no_hand_count = 0
STABLE_THRESHOLD = 12
NO_HAND_THRESHOLD = 30

cap = cv2.VideoCapture(0)

def gen_frames():
    global current_text, last_pred, stable_count, no_hand_count
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            no_hand_count = 0
            hand = res.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            feature = np.array([lm for lm in sum([[lm.x, lm.y, lm.z] for lm in hand.landmark], [])]).reshape(1, -1)
            probs = model.predict(feature, verbose=0)
            idx = np.argmax(probs)
            pred_label = lb.classes_[idx]
            conf = probs[0, idx]

            if last_pred == pred_label:
                stable_count += 1
            else:
                last_pred = pred_label
                stable_count = 1

            if stable_count >= STABLE_THRESHOLD:
                if len(current_text) == 0 or current_text[-1] != pred_label:
                    current_text += pred_label
                stable_count = 0
                last_pred = None

            cv2.putText(frame, f"{pred_label} ({conf:.2f})", (10,80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,0),3)
        else:
            no_hand_count += 1
            stable_count = 0
            last_pred = None
            if no_hand_count == NO_HAND_THRESHOLD:
                if len(current_text) > 0 and current_text[-1] != " ":
                    current_text += " "
                no_hand_count = 0

        # Overlay text
        display_text = current_text[-60:]
        cv2.rectangle(frame, (0,h-120), (w,h), (0,0,0), -1)
        cv2.putText(frame, display_text, (10,h-40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255),3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- API Routes ---
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_text')
def get_text():
    global current_text
    return jsonify({"text": current_text})

@app.route('/clear_text', methods=['POST'])
def clear_text():
    global current_text
    current_text = ""
    return jsonify({"status":"ok"})

@app.route('/backspace', methods=['POST'])
def backspace():
    global current_text
    current_text = current_text[:-1]
    return jsonify({"status":"ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
