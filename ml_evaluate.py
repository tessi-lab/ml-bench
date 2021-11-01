from typing import Any

import numpy as np
import tensorflow as tf
from keras.models import Model
from tensorflow.keras.models import load_model


def load_mnist_model(filename: str) -> Any:
    return load_model(filename)


def normalize_img(image: np.ndarray):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.


def predict(model: Model, image: np.ndarray) -> int:
    img = normalize_img(image)
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, -1)
    return model.predict(img).argmax()