"""
Pytest configuration file for signxai
"""
import pytest
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16

@pytest.fixture(scope="session")
def dummy_vgg16_model():
    """
    Create a VGG16 model for testing
    """
    model = VGG16(weights=None, include_top=True, classes=1000, input_shape=(224, 224, 3))
    # Remove softmax from the final layer
    model.layers[-1].activation = None
    return model

@pytest.fixture(scope="session")
def dummy_input():
    """
    Create dummy input for testing
    """
    # Create a dummy RGB input of shape (224, 224, 3)
    np.random.seed(42)
    return np.random.rand(224, 224, 3)

@pytest.fixture(scope="session")
def dummy_batch_input():
    """
    Create dummy batch input for testing
    """
    # Create a dummy batch of 5 RGB inputs of shape (5, 224, 224, 3)
    np.random.seed(42)
    return np.random.rand(5, 224, 224, 3)

@pytest.fixture(scope="session")
def dummy_mnist_model():
    """
    Create a simple model for MNIST dataset
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    # Remove softmax from the final layer
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

@pytest.fixture(scope="session")
def dummy_mnist_input():
    """
    Create dummy input for MNIST testing
    """
    # Create a dummy grayscale input of shape (28, 28, 1)
    np.random.seed(42)
    return np.random.rand(28, 28, 1)