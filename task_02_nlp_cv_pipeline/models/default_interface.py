from abc import ABC, abstractmethod

class ModelInterface(ABC):
    """
    Abstract base class (ABC) that defines the interface for machine learning models.

    This interface enforces a common structure for all models by requiring the implementation
    of four essential methods: training, prediction, saving, and loading model weights.
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model using the provided training dataset.

        :param X_train: Feature set for training.
        :param y_train: Target labels corresponding to X_train.
        :return: None
        """
        pass

    @abstractmethod
    def predict(self, X_test):
        """
        Generates predictions based on the trained model.

        :param X_test: Feature set for testing.
        :return: Predicted labels or values.
        """
        pass

    @abstractmethod
    def save_weights(self, path: str):
        """
        Saves the trained model weights to a specified file.

        :param path: File path where the model weights should be saved.
        :return: None
        """
        pass

    @abstractmethod
    def load_weights(self, path: str):
        """
        Loads the model weights from a specified file.

        :param path: File path from which the model weights should be loaded.
        :return: None
        """
        pass