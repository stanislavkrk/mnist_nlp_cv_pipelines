import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .mnist_classifier_interface import MnistClassifierInterface
import numpy as np

class FFNNModel(MnistClassifierInterface):
    """
    Feed-Forward Neural Network (FFNN) model for MNIST digit classification.

    This model consists of a fully connected hidden layer followed by an output layer with softmax activation.
    It is trained using sparse categorical cross-entropy loss and optimized with Adam.
    """

    def __init__(self, input_size=784, hidden_size=128, output_size=10,
                 learning_rate=0.001, epochs=10, batch_size=32, random_state=42):
        """
        Initializes the FFNN model with specified hyperparameters.

        :param input_size: The number of input features (flattened 28x28 MNIST images).
        :param hidden_size: The number of neurons in the hidden layer.
        :param output_size: The number of output classes (10 digits: 0-9).
        :param learning_rate: The learning rate for the Adam optimizer.
        :param epochs: The number of training epochs.
        :param batch_size: The number of samples per training batch.
        :param random_state: The seed for TensorFlow's random number generator.
        """

        tf.random.set_seed(random_state)

        self.epochs = epochs
        self.batch_size = batch_size

        # Define the FFNN architecture
        self.model = keras.Sequential([
            layers.Dense(hidden_size, activation="relu", input_shape=(input_size,)), # Fully connected hidden layer
            layers.Dense(output_size, activation="softmax") # Output layer with softmax activation
        ])

        # Compile the model with Adam optimizer and sparse categorical cross-entropy loss
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                           loss="sparse_categorical_crossentropy",
                           metrics=["accuracy"])

    def train(self, X_train, y_train):
        """
        Trains the FFNN model using the provided MNIST dataset.

        :param X_train: A NumPy array of shape (num_samples, input_size) containing the training images.
        :param y_train: A NumPy array of shape (num_samples,) containing the corresponding labels.
        :return: None
        """
        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1)

    def predict(self, X_test):
        """
        Generates predictions using the trained FFNN model.

        :param X_test: A NumPy array of shape (num_samples, input_size) containing the test images.
        :return: A NumPy array of predicted class labels (digits 0-9).
        """
        predictions = self.model.predict(X_test)

        # Convert softmax outputs to class labels
        return np.argmax(predictions, axis=1)