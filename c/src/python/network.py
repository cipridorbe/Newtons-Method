import tensorflow as tf
from tensorflow.keras import layers, models, utils
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load and preprocess the digits dataset
digits = load_digits()
X = digits.data  # shape: (n_samples, 64) since images are 8x8
y = digits.target

# Scale the features for better training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert labels to one-hot encoded vectors (10 classes: digits 0-9)
y_categorical = utils.to_categorical(y, num_classes=10)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42)

# Build the model: input layer is defined by the input shape of the first Dense layer.
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(64,)),  # Input layer + 1st hidden layer
    layers.Dense(64, activation='relu'),                      # 2nd hidden layer
    layers.Dense(32, activation='relu'),                      # 3rd hidden layer
    layers.Dense(10, activation='softmax')                    # Output layer for 10 classes
])

# Compile the model using an actual optimizer instance
model.compile(optimizer=Adam(learning_rate=0.001),  # Correct optimizer usage
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model for 100 epochs
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")
