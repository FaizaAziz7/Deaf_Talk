# prediction_service.py
import joblib
import numpy as np
import tensorflow as tf

# Load model and scaler
MODEL_PATH = "models/urdu_sign_model.tflite"
SCALER_PATH = "models/scaler.pkl"
LABELS_PATH = "models/labels.txt"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

scaler = joblib.load(SCALER_PATH)

with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(features: list):
    # Scale input
    X = scaler.transform([features]).astype(np.float32)

    # Run TFLite inference
    interpreter.set_tensor(input_details[0]["index"], X)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]["index"])[0]

    # Get best label
    predicted_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    return {"label": labels[predicted_index], "confidence": confidence}
