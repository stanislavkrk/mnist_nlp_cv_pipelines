import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .mnist_classifier_interface import MnistClassifierInterface
import numpy as np

class CNNModel(MnistClassifierInterface):
    """
    Convolutional Neural Network (CNN) for MNIST classification.

    This class implements a CNN model using TensorFlow/Keras. It consists of convolutional
    and max-pooling layers, followed by fully connected layers, for classifying MNIST digits.
    """

    def __init__(self, learning_rate=0.001, epochs=10, batch_size=32, random_state=42):
        """
        Initializes the CNN model with specified hyperparameters.

        :param learning_rate: Learning rate for Adam optimizer (default: 0.001).
        :param epochs: Number of training epochs (default: 10).
        :param batch_size: Batch size for training (default: 32).
        :param random_state: Random seed for reproducibility (default: 42).
        """
        # Set random seed for reproducibility
        tf.random.set_seed(random_state)

        self.epochs = epochs
        self.batch_size = batch_size

        # Define the CNN architecture
        self.model = keras.Sequential([
            layers.Conv2D(32, kernel_size=(2, 3), activation="relu", input_shape=(28, 28, 1)),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(10, activation="softmax")
        ])

        # Compile the model with Adam optimizer and sparse categorical cross-entropy loss
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                           loss="sparse_categorical_crossentropy",
                           metrics=["accuracy"])

    def train(self, X_train, y_train):
        """
        Trains the CNN model on the provided MNIST dataset.

        :param X_train: Training images as a NumPy array (shape: [num_samples, 28, 28, 1]).
        :param y_train: Corresponding labels as a NumPy array (shape: [num_samples]).
        :return: None
        """
        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1)

    def predict(self, X_test):
        """
        Generates predictions using the trained CNN model.

        :param X_test: Test images as a NumPy array (shape: [num_samples, 28, 28, 1]).
        :return: Predicted class labels as a NumPy array.
        """
        predictions = self.model.predict(X_test)

        # Convert softmax outputs to class labels
        return np.argmax(predictions, axis=1)