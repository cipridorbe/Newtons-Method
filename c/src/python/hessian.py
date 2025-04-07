import os
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, utils
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

EPOCHS = 50

# Load and preprocess data
digits = load_digits()
X = digits.data
y = digits.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_categorical = utils.to_categorical(y, num_classes=10)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42)

# Define model builder
def build_model():
    return models.Sequential([
        layers.Input(shape=(64,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

loss_fn = tf.keras.losses.CategoricalCrossentropy()

# Newton's method step
@tf.function
def newton_step(model, X_batch, y_batch, damping=10.0, step_size=0.05):
    with tf.GradientTape(persistent=True) as tape2:
        with tf.GradientTape() as tape1:
            preds = model(X_batch)
            loss = loss_fn(y_batch, preds)
        grads = tape1.gradient(loss, model.trainable_variables)

    updates = []
    for grad, var in zip(grads, model.trainable_variables):
        if grad is None:
            continue
        hessian = tape2.jacobian(grad, var)

        grad_1d = tf.reshape(grad, [-1])
        hessian_2d = tf.reshape(hessian, [grad_1d.shape[0], grad_1d.shape[0]])
        hessian_2d += tf.eye(hessian_2d.shape[0]) * damping

        try:
            delta = tf.linalg.solve(hessian_2d, grad_1d[:, tf.newaxis])
            delta_reshaped = tf.reshape(delta, var.shape)
            updates.append((var, delta_reshaped))
        except tf.errors.InvalidArgumentError:
            print("Skipping update due to singular Hessian.")
            continue

    del tape1
    del tape2

    for var, delta in updates:
        var.assign_sub(step_size * delta)

    return loss

# Training with Newton
def train_newton(model, X_train, y_train, epochs=EPOCHS, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
    history = {"loss": [], "acc": []}
    for epoch in range(epochs):
        epoch_loss = []
        correct = total = 0
        for X_batch, y_batch in dataset:
            X_batch = tf.cast(X_batch, tf.float32)
            y_batch = tf.cast(y_batch, tf.float32)
            loss = newton_step(model, X_batch, y_batch)
            epoch_loss.append(loss.numpy())
            preds = model(X_batch)
            correct += tf.reduce_sum(tf.cast(tf.equal(tf.argmax(preds, 1), tf.argmax(y_batch, 1)), tf.float32)).numpy()
            total += X_batch.shape[0]
        history["loss"].append(np.mean(epoch_loss))
        history["acc"].append(correct / total)
        print(f"Newton Epoch {epoch+1}: Loss={history['loss'][-1]:.4f}, Accuracy={history['acc'][-1]:.4f}")
    return history

# Adam
def train_adam(X_train, y_train, X_test, y_test):
    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=EPOCHS, batch_size=32, verbose=1)
    return history, model

# SGD
def train_sgd(X_train, y_train, X_test, y_test):
    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=EPOCHS, batch_size=32, verbose=1)
    return history, model

# Utilities for model saving/loading and CSV

def save_model(model, name):
    model.save(f"{name}_model.h5")

def load_model(name):
    return tf.keras.models.load_model(f"{name}_model.h5")

def write_history_to_csv(histories, filename="results.csv"):
    fieldnames = ["epoch"]
    for model_name in histories:
        fieldnames += [f"{model_name}_loss", f"{model_name}_accuracy"]

    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(EPOCHS):
            row = {"epoch": epoch + 1}
            for model_name, history in histories.items():
                row[f"{model_name}_loss"] = history["loss"][epoch]
                row[f"{model_name}_accuracy"] = history["acc"][epoch]
            writer.writerow(row)

def prompt_load_or_train(model_name):
    file_path = f"{model_name}_model.h5"
    if os.path.exists(file_path):
        response = input(f"Saved model '{file_path}' found. Load instead of training? (y/n): ").strip().lower()
        return response == "y"
    return False

# ---------- Training ----------

histories = {}

if prompt_load_or_train("newton"):
    newton_model = load_model("newton")
    newton_history = {"loss": [0]*EPOCHS, "acc": [0]*EPOCHS}
else:
    newton_model = build_model()
    newton_history = train_newton(newton_model, X_train, y_train)
    save_model(newton_model, "newton")
histories["newton"] = newton_history
newton_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
newton_loss, newton_acc = newton_model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal Newton Test Accuracy: {newton_acc:.4f}")

if prompt_load_or_train("adam"):
    adam_model = load_model("adam")
    adam_history = {"loss": [0]*EPOCHS, "acc": [0]*EPOCHS}
else:
    adam_history_raw, adam_model = train_adam(X_train, y_train, X_test, y_test)
    adam_history = {"loss": adam_history_raw.history["loss"], "acc": adam_history_raw.history["accuracy"]}
    save_model(adam_model, "adam")
histories["adam"] = adam_history
adam_loss, adam_acc = adam_model.evaluate(X_test, y_test, verbose=0)
print(f"Final Adam Test Accuracy: {adam_acc:.4f}")

if prompt_load_or_train("sgd"):
    sgd_model = load_model("sgd")
    sgd_history = {"loss": [0]*EPOCHS, "acc": [0]*EPOCHS}
else:
    sgd_history_raw, sgd_model = train_sgd(X_train, y_train, X_test, y_test)
    sgd_history = {"loss": sgd_history_raw.history["loss"], "acc": sgd_history_raw.history["accuracy"]}
    save_model(sgd_model, "sgd")
histories["sgd"] = sgd_history
sgd_loss, sgd_acc = sgd_model.evaluate(X_test, y_test, verbose=0)
print(f"Final SGD Test Accuracy: {sgd_acc:.4f}")

write_history_to_csv(histories)

# ---------- Plot ----------

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(histories["newton"]["loss"], label="Newton")
plt.plot(histories["adam"]["loss"], label="Adam")
plt.plot(histories["sgd"]["loss"], label="SGD")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(histories["newton"]["acc"], label="Newton")
plt.plot(histories["adam"]["acc"], label="Adam")
plt.plot(histories["sgd"]["acc"], label="SGD")
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

# ---------- Show Random Test Predictions ----------

num_samples = 10
indices = random.sample(range(len(X_test)), num_samples)
X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
preds_newton = tf.argmax(newton_model(X_test_tensor), axis=1).numpy()
preds_adam = tf.argmax(adam_model(X_test_tensor), axis=1).numpy()
preds_sgd = tf.argmax(sgd_model(X_test_tensor), axis=1).numpy()
true_labels = np.argmax(y_test, axis=1)

plt.figure(figsize=(15, 6))
for i, idx in enumerate(indices):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[idx].reshape(8, 8), cmap='gray')
    plt.axis('off')
    plt.title(
        f"True: {true_labels[idx]}\n"
        f"Newton: {preds_newton[idx]}\n"
        f"Adam: {preds_adam[idx]}\n"
        f"SGD: {preds_sgd[idx]}"
    )
plt.suptitle("Random Test Samples with Predictions")
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()
