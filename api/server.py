import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import pickle
import logging
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
from google.cloud import speech_v1p1beta1 as speech
import os
import io

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Server Section ---
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8081",
    "http://192.168.1.X:8081",  # Replace with your IP
    "*"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

conn = sqlite3.connect('user_data.db', check_same_thread=False)
conn.execute('CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, features TEXT, difficulty INT)')
conn.execute('CREATE TABLE IF NOT EXISTS pronunciation_scores (id INTEGER PRIMARY KEY AUTOINCREMENT, audio_file TEXT, score REAL)')
logger.info("SQLite database initialized")

# Load difficulty model
diff_interpreter = tf.lite.Interpreter(model_path="adaptive_learning_model.tflite")
diff_interpreter.allocate_tensors()
diff_input_details = diff_interpreter.get_input_details()
diff_output_details = diff_interpreter.get_output_details()
logger.info("Difficulty model loaded into interpreter")

with open("scaler.pkl", "rb") as f:
    diff_scaler = pickle.load(f)
logger.info("Difficulty scaler loaded")

# Load pronunciation model
pron_interpreter = tf.lite.Interpreter(model_path="pronunciation_model.tflite")
pron_interpreter.allocate_tensors()
pron_input_details = pron_interpreter.get_input_details()
pron_output_details = pron_interpreter.get_output_details()
logger.info("Pronunciation model loaded into interpreter")

with open("pron_scaler.pkl", "rb") as f:
    pron_scaler = pickle.load(f)
logger.info("Pronunciation scaler loaded")

class UserData(BaseModel):
    features: list[float]

def predict_difficulty(features):
    normalized_features = diff_scaler.transform([features]).astype(np.float32)
    diff_interpreter.set_tensor(diff_input_details[0]['index'], normalized_features)
    diff_interpreter.invoke()
    output_data = diff_interpreter.get_tensor(diff_output_details[0]['index'])
    return np.argmax(output_data)

def predict_pronunciation(features):
    # Simulated audio features (replace with real extraction later)
    normalized_features = pron_scaler.transform([features]).astype(np.float32)
    pron_interpreter.set_tensor(pron_input_details[0]['index'], normalized_features)
    pron_interpreter.invoke()
    score = pron_interpreter.get_tensor(pron_output_details[0]['index'])[0]
    return float(score)

@app.post("/predict")
async def get_prediction(data: UserData):
    if len(data.features) != 11:
        raise HTTPException(status_code=400, detail="Exactly 11 features required")
    try:
        difficulty = predict_difficulty(data.features)
        conn.execute('INSERT INTO predictions (features, difficulty) VALUES (?, ?)', 
                    (str(data.features), difficulty))
        conn.commit()
        logger.info(f"Predicted difficulty: {difficulty}")
        return {"difficulty": int(difficulty)}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/pronunciation")
async def analyze_pronunciation(audio_file: UploadFile = File(...)):
    try:
        # Google Cloud Speech-to-Text for transcription
        client = speech.SpeechClient()
        audio_content = await audio_file.read()
        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
        )
        response = client.recognize(config=config, audio=audio)
        transcript = response.results[0].alternatives[0].transcript if response.results else "Unknown"

        # Simulated audio features (replace with real analysis, e.g., from librosa)
        audio_features = np.random.rand(5)  # Placeholder: pitch, clarity, speed, accuracy, fluency
        score = predict_pronunciation(audio_features)
        
        # Log to SQLite
        conn.execute('INSERT INTO pronunciation_scores (audio_file, score) VALUES (?, ?)', 
                    (audio_file.filename, score))
        conn.commit()
        logger.info(f"Pronunciation score: {score}, Transcript: {transcript}")
        return {"score": score, "transcript": transcript}
    except Exception as e:
        logger.error(f"Pronunciation error: {e}")
        raise HTTPException(status_code=500, detail="Pronunciation analysis failed")

@app.get("/")
async def root():
    return {"message": "Model server is running"}

if __name__ == "__main__":
    logger.info("Training complete, starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)