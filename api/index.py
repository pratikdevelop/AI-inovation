import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate dataset
np.random.seed(42)
num_samples = 200
X = np.random.rand(num_samples, 11)
y = np.random.randint(0, 3, num_samples)

# Simulate nuanced data
for i in range(num_samples):
    if y[i] == 2:  # Advanced
        X[i] = np.clip(X[i] + np.random.uniform(0.2, 0.4), 0, 1)
    elif y[i] == 0:  # Beginner
        X[i] = np.clip(X[i] - np.random.uniform(0.2, 0.4), 0, 1)
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
print(f"Test accuracy: {test_accuracy:.4f}")

# Test a sample prediction
sample_user = np.array([[0.8, 0.7, 0.6, 0.5, 0.4, 0.9, 0.85, 0.2, 0.75, 0.9, 0.65]])
sample_user = scaler.transform(sample_user)
prediction = model.predict(sample_user)
difficulty = np.argmax(prediction)
print(f"Predicted difficulty level for sample user: {difficulty} (0=beginner, 1=intermediate, 2=advanced)")

# Convert to TensorFlow Lite with quantization toggle
use_full_integer_quantization = True  # Set to True for ~14 KB, False for ~30 KB with higher accuracy
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
if use_full_integer_quantization:
    converter.target_spec.supported_types = [tf.int8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    def representative_dataset():
        for _ in range(100):
            yield [np.random.rand(1, 11).astype(np.float32)]
    converter.representative_dataset = representative_dataset
tflite_model = converter.convert()

# Save the model
with open("adaptive_learning_model.tflite", "wb") as f:
    f.write(tflite_model)

# Check model size
import os
model_size = os.path.getsize("adaptive_learning_model.tflite") / 1024  # Size in KB
print(f"Model size: {model_size:.2f} KB")

print("Model trained and saved as adaptive_learning_model.tflite")