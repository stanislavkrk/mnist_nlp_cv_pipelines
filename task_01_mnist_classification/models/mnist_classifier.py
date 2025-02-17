from .mnist_classifier_interface import MnistClassifierInterface
from .model_random_forest import RandomForestModel
from .model_ffnn import FFNNModel
from .model_cnn import CNNModel

class MnistClassifier:
    """
    A unified classifier for handling different MNIST classification models.

    This class serves as a wrapper for multiple ML models (Random Forest, FFNN, CNN),
    allowing seamless training and inference on MNIST dataset.
    """
    def __init__(self, model_type='default'):
        """
        Initializes the MNIST classifier by selecting the appropriate model.

        :param model_type: Type of model to use ('rf' for Random Forest, 'nn' for FFNN, 'cnn' for CNN).
        :raises ValueError: If an unsupported model type is provided.
        :raises TypeError: If the selected model does not implement MnistClassifierInterface.
        """
        match model_type:
            case "rf":
                self.model = RandomForestModel()
            case "nn":
                self.model = FFNNModel()
            case "cnn":
                self.model = CNNModel()
            case _:
                raise ValueError(f"Unknown model type: {model_type}. Use 'rf', 'nn', or 'cnn'.")

        # Ensure that the selected model implements the required interface
        if not isinstance(self.model, MnistClassifierInterface):
            raise TypeError(f"This model {type(self.model).__name__} doesn`t realise MnistClassifierInterface!")


    def train(self, X_train, y_train):
        """
        Trains the selected MNIST model on the provided dataset.

        :param X_train: Training feature set (image data).
        :param y_train: Corresponding labels for training data.
        :return: None
        """
        self.model.train(X_train, y_train)

    def predict(self, X_test):
        """
        Generates predictions using the trained MNIST model.

        :param X_test: Feature set (image data) for testing.
        :return: Predicted class labels.
        """
        return self.model.predict(X_test)