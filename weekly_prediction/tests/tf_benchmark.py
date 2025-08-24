import tensorflow as tf
from tensorflow.keras import mixed_precision
import time

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

# Enable mixed precision
mixed_precision.set_global_policy("mixed_float16")

(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_train = x_train[..., None]  # add channel dimension

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax", dtype="float32")  # keep softmax float32
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def train_on(device_name):
    with tf.device(device_name):
        model = build_model()
        start = time.time()
        model.fit(x_train, y_train, epochs=3, batch_size=2048, verbose=0)
        end = time.time()
        print(f"Time on {device_name}: {end - start:.2f} sec")

train_on("/CPU:0")
if tf.config.list_physical_devices("GPU"):
    train_on("/GPU:0")
