import tensorflow as tf
from tensorflow.keras import layers, models, utils
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

EPOCHS = 20

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

# Training with Adam
def train_adam(X_train, y_train, X_test, y_test):
    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=EPOCHS, batch_size=32, verbose=1)
    return history, model

# Training with plain SGD (gradient descent)
def train_sgd(X_train, y_train, X_test, y_test):
    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=EPOCHS, batch_size=32, verbose=1)
    return history, model

# ---------- Training ----------

# Newton's Method
newton_model = build_model()
newton_history = train_newton(newton_model, X_train, y_train)
newton_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
newton_loss, newton_acc = newton_model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal Newton Test Accuracy: {newton_acc:.4f}")

# Adam
adam_history, adam_model = train_adam(X_train, y_train, X_test, y_test)
adam_loss, adam_acc = adam_model.evaluate(X_test, y_test, verbose=0)
print(f"Final Adam Test Accuracy: {adam_acc:.4f}")

# SGD
sgd_history, sgd_model = train_sgd(X_train, y_train, X_test, y_test)
sgd_loss, sgd_acc = sgd_model.evaluate(X_test, y_test, verbose=0)
print(f"Final SGD Test Accuracy: {sgd_acc:.4f}")

# ---------- Plot ----------

plt.figure(figsize=(14, 6))

# Loss
plt.subplot(1, 2, 1)
plt.plot(newton_history["loss"], label="Newton")
plt.plot(adam_history.history["loss"], label="Adam")
plt.plot(sgd_history.history["loss"], label="SGD")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(newton_history["acc"], label="Newton")
plt.plot(adam_history.history["accuracy"], label="Adam")
plt.plot(sgd_history.history["accuracy"], label="SGD")
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
