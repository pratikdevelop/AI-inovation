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

# --- Training Section for Difficulty Model ---
np.random.seed(42)
num_samples = 200
X = np.random.rand(num_samples, 11) * 0.5
y = np.random.randint(0, 3, num_samples)

for i in range(num_samples):
    if y[i] == 0:  # Beginner
        X[i] = np.zeros(11)
    elif y[i] == 2:  # Advanced
        X[i] = np.clip(X[i] + np.random.uniform(0.3, 0.5), 0, 1)
    else:  # Intermediate
        X[i] = np.clip(X[i] + np.random.uniform(-0.1, 0.1), 0, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(11,), kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

initial_learning_rate = 0.001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=1000, decay_rate=0.9
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    verbose=1
)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
logger.info(f"Difficulty model test accuracy: {test_accuracy:.4f}")

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
logger.info("Scaler saved as scaler.pkl")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
def representative_dataset():
    for _ in range(100):
        yield [np.random.rand(1, 11).astype(np.float32)]
converter.representative_dataset = representative_dataset
tflite_model = converter.convert()

with open("adaptive_learning_model.tflite", "wb") as f:
    f.write(tflite_model)
logger.info("Difficulty model saved as adaptive_learning_model.tflite")

# --- Training Section for Pronunciation Model (Simplified Example) ---
# Simulated pronunciation data: 5 features (e.g., pitch, clarity, speed, accuracy, fluency), score 0-1
num_pron_samples = 200
X_pron = np.random.rand(num_pron_samples, 5)  # Replace with real audio features later
y_pron = np.random.uniform(0, 1, num_pron_samples)  # Continuous score 0-1

X_pron_train, X_pron_test, y_pron_train, y_pron_test = train_test_split(X_pron, y_pron, test_size=0.2, random_state=42)
pron_scaler = StandardScaler()
X_pron_train = pron_scaler.fit_transform(X_pron_train)
X_pron_test = pron_scaler.transform(X_pron_test)

pron_model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(5,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')  # Score 0-1
])

pron_model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mae']
)

pron_model.fit(X_pron_train, y_pron_train, epochs=50, batch_size=16, validation_data=(X_pron_test, y_pron_test), verbose=1)
pron_test_loss, pron_test_mae = pron_model.evaluate(X_pron_test, y_pron_test)
logger.info(f"Pronunciation model test MAE: {pron_test_mae:.4f}")

with open("pron_scaler.pkl", "wb") as f:
    pickle.dump(pron_scaler, f)
logger.info("Pronunciation scaler saved as pron_scaler.pkl")

pron_converter = tf.lite.TFLiteConverter.from_keras_model(pron_model)
pron_converter.optimizations = [tf.lite.Optimize.DEFAULT]
pron_converter.representative_dataset = lambda: ([np.random.rand(1, 5).astype(np.float32)] for _ in range(100))
pron_tflite_model = pron_converter.convert()

with open("pronunciation_model.tflite", "wb") as f:
    f.write(pron_tflite_model)
logger.info("Pronunciation model saved as pronunciation_model.tflite")