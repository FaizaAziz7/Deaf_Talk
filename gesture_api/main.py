from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib
import pandas as pd
import tempfile

# ---------------- Paths ----------------
TFLITE_PATH = "models/model_conv_lstm_float16.tflite"
SCALER_PATH = "models/scaler"
LABEL_MAP_FILE = "models/label_map_inv.csv"

# ---------------- Load artifacts ----------------
scaler = joblib.load(SCALER_PATH)
df_labels = pd.read_csv(LABEL_MAP_FILE)
label_map = {i: name for i, name in enumerate(df_labels["class_name"].tolist())}

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------------- MediaPipe Hands ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.6)

# ---------------- FastAPI app ----------------
app = FastAPI()

# Allow Flutter (mobile/web) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later replace with your Flutter app domain/IP
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_keypoints(results):
    keypoints = []
    for hand_landmarks in results.multi_hand_landmarks or []:
        for lm in hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    if len(keypoints) < 126:
        keypoints.extend([0] * (126 - len(keypoints)))
    return np.array(keypoints, dtype=np.float32)

def predict_gesture(keypoints):
    X_scaled = scaler.transform([keypoints])
    X_scaled = X_scaled.astype(np.float32)
    X_in = X_scaled.reshape((1, 126, 1))
    interpreter.set_tensor(input_details[0]['index'], X_in)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    top_indices = np.argsort(output)[::-1][:3]
    predictions = [
        {"label": label_map[i], "confidence": float(output[i])}
        for i in top_indices
    ]
    return predictions
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save file temporarily
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(await file.read())
    temp.close()

    filename = file.filename.lower()

    # ---------- 1) CHECK IMAGE ----------
    if any(filename.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
        img = cv2.imread(temp.name)
        if img is None:
            return {"predictions": [], "error": "Could not read image"}

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return {"predictions": []}

        keypoints = extract_keypoints(results)
        predictions = predict_gesture(keypoints)
        return {"predictions": predictions}

    # ---------- 2) CHECK VIDEO ----------
    elif filename.endswith(".mp4"):
        cap = cv2.VideoCapture(temp.name)
        if not cap.isOpened():
            return {"predictions": [], "error": "Could not open video"}

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mid_frame_index = frame_count // 2

        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_index)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return {"predictions": [], "error": "Could not read frame from video"}

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return {"predictions": []}

        keypoints = extract_keypoints(results)
        predictions = predict_gesture(keypoints)
        return {"predictions": predictions}

    # ---------- 3) UNSUPPORTED FILE ----------
    else:
        return {"error": "Unsupported file type. Please upload .jpg, .jpeg, .png or .mp4"}
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
