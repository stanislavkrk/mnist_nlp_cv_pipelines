import numpy as np
from tensorflow.keras.utils import to_categorical

class Preprocessor:
    """
    Data preprocessor for MNIST classification.

    This class preprocesses input images and labels based on the selected model type:
    - 'rf' (Random Forest): No normalization, no one-hot encoding, reshapes to a flat vector.
    - 'nn' (Feed-Forward Neural Network): Normalizes images, no one-hot encoding, reshapes to a flat vector.
    - 'cnn' (Convolutional Neural Network): Normalizes images, no one-hot encoding, reshapes to CNN format (28x28x1).
    """

    def __init__(self, model_type="default"):
        """
        Initializes the preprocessor with the appropriate settings based on the model type.

        :param model_type: The type of model to preprocess data for ('rf', 'nn', or 'cnn').
        :raises ValueError: If an unsupported model type is provided.
        """
        match model_type:
            # Random Forest
            case "rf":
                self.normalize = False
                self.one_hot = False
                self.reshape_mode = "flat"
            # Feed-Forward Neural Network
            case "nn":
                self.normalize = True
                self.one_hot = False
                self.reshape_mode = "flat"
            # Convolutional Neural Network
            case "cnn":
                self.normalize = True
                self.one_hot = False
                self.reshape_mode = "cnn"
            case _:
                raise ValueError(f"Unknown model type: {model_type}. Use 'rf', 'nn', or 'cnn'.")

    def process(self, X, y):
        """
        Preprocesses input images and labels based on the initialized model settings.

        - Normalizes pixel values to [0, 1] if required.
        - Reshapes images into either flat vectors or CNN-compatible format.
        - Converts labels to one-hot encoding if required.

        :param X: A NumPy array of shape (num_samples, 28, 28) containing grayscale MNIST images.
        :param y: A NumPy array of shape (num_samples,) containing integer class labels (0-9).
        :return: A tuple (X_processed, y_processed) where:
                 - X_processed is a NumPy array with appropriate reshaping and normalization.
                 - y_processed is either a categorical array (if one-hot encoding is applied) or the original labels.
        """
        if self.normalize:
            X = X / 255.0

        if self.reshape_mode == 'flat':
            X = X.reshape(X.shape[0], -1)
        elif self.reshape_mode == 'cnn':
            X = X.reshape(X.shape[0], 28, 28, 1)

        if self.one_hot:
            y = to_categorical(y, num_classes=10)

        return X, y