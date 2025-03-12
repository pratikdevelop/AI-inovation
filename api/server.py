import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import logging
import uvicorn
from fastapi.middleware.cors import CORSMiddleware


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Training Section ---
np.random.seed(42)
num_samples = 200
X = np.random.rand(num_samples, 11)
y = np.random.randint(0, 3, num_samples)

# Simulate nuanced data
# Adjust training data
X = np.random.rand(num_samples, 11) * 0.5  # Lower range for more realistic zeros
for i in range(num_samples):
    if y[i] == 0:  # Beginner
        X[i] = np.zeros(11)  # Explicitly set zeros for beginner
    elif y[i] == 2:  # Advanced
        X[i] = np.clip(X[i] + np.random.uniform(0.3, 0.5), 0, 1)
    else:  # Intermediate
        X[i] = np.clip(X[i] + np.random.uniform(-0.1, 0.1), 0, 1)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(11,), kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

# Learning rate scheduler
initial_learning_rate = 0.001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=1000, decay_rate=0.9
)

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
logger.info(f"Test accuracy: {test_accuracy:.4f}")

# Save the scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
logger.info("Scaler saved as scaler.pkl")

# Convert to TensorFlow Lite with full integer quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Remove INT8 constraints
# converter.target_spec.supported_types = [tf.int8]
# converter.inference_input_type = tf.int8
# converter.inference_output_type = tf.int8
def representative_dataset():
    for _ in range(100):
        yield [np.random.rand(1, 11).astype(np.float32)]
converter.representative_dataset = representative_dataset
tflite_model = converter.convert()

# Save the .tflite model
with open("adaptive_learning_model.tflite", "wb") as f:
    f.write(tflite_model)
logger.info("Model saved as adaptive_learning_model.tflite")

# # --- Server Section ---
app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:8081",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the .tflite model
interpreter = tf.lite.Interpreter(model_path="adaptive_learning_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
logger.info("Model loaded into interpreter")

# Get quantization parameters
input_scale = input_details[0]['quantization'][0]
input_zero_point = input_details[0]['quantization'][1]
logger.info(f"Input scale: {input_scale}, Zero point: {input_zero_point}")

# Load the scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
logger.info("Scaler loaded for server")
logger.info(f"Scaler mean: {scaler.mean_}")
logger.info(f"Scaler std: {scaler.scale_}")

# Define input schema
class UserData(BaseModel):
    features: list[float]

# Normalize and quantize input
def normalize_and_quantize_input(features):
    # Log raw input
    logger.info(f"Raw input features: {features}")
    # Normalize with scaler
    normalized_features = scaler.transform([features])
    logger.info(f"Normalized features: {normalized_features}")
    # Quantize to INT8
    quantized_features = np.round(normalized_features / input_scale + input_zero_point).astype(np.int8)
    logger.info(f"Quantized features: {quantized_features}")
    return quantized_features

# Predict difficulty
def predict_difficulty(features):
    normalized_features = scaler.transform([features]).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], normalized_features)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(output_data)

@app.post("/predict")
async def get_prediction(data: UserData):
    if len(data.features) != 11:
        raise HTTPException(status_code=400, detail="Exactly 11 features required")
    try:
        difficulty = predict_difficulty(data.features)
        logger.info(f"Predicted difficulty: {difficulty}")
        return {"difficulty": int(difficulty)}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.get("/")
async def root():
    return {"message": "Model server is running"}

if __name__ == "__main__":
    logger.info("Training complete, starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)