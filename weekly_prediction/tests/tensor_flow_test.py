import tensorflow as tf
from tensorflow import keras

print("TF version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

# Simple test model
model = keras.Sequential([keras.layers.Dense(128, activation='relu'),
                          keras.layers.Dense(10, activation='softmax')])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
print("Accelerator backend:", tf.config.list_physical_devices("GPU"))


print("TensorFlow:", tf.__version__)
print("Keras:", tf.keras.__version__)
print("Available devices:", tf.config.list_physical_devices())

print("GPUs:", tf.config.list_physical_devices("GPU"))
