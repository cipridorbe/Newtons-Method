import tensorflow as tf
from tensorflow import keras

# Define a simple model
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(5,)),
    keras.layers.Dense(1)
])

# Generate dummy input data
x = tf.random.normal((1, 5))  # Batch size of 1, input size of 5

# Define a loss function
with tf.GradientTape(persistent=True) as tape2:
    with tf.GradientTape(persistent=True) as tape1:
        y_pred = model(x)  # Forward pass
        loss = tf.reduce_sum(y_pred)  # Simple sum loss function
    
    # Compute first-order gradients (Jacobian)
    grads = tape1.gradient(loss, model.trainable_variables)

# Compute the Hessian for each trainable variable
hessians = [tape2.jacobian(g, w) for g, w in zip(grads, model.trainable_variables)]

# Print Hessian shapes
for i, h in enumerate(hessians):
    print(f"Hessian for variable {i}: shape {h.shape}")

# Clean up tapes to free memory
del tape1
del tape2
