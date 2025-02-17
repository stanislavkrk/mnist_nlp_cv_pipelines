from sklearn.ensemble import RandomForestClassifier
from .mnist_classifier_interface import MnistClassifierInterface

class RandomForestModel(MnistClassifierInterface):
    """
    Random Forest model for MNIST digit classification.

    This model is based on the RandomForestClassifier from scikit-learn.
    It is a non-neural approach to MNIST classification, leveraging decision trees.
    """

    def __init__(self, n_estimators=100, random_state=42):
        """
        Initializes the Random Forest model with specified hyperparameters.

        :param n_estimators: The number of trees in the forest.
        :param random_state: The seed for random number generation to ensure reproducibility.
        """
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)


    def train(self, X_train, y_train):
        """
        Trains the Random Forest model using the provided MNIST dataset.

        :param X_train: A NumPy array of shape (num_samples, num_features) containing the training images.
        :param y_train: A NumPy array of shape (num_samples,) containing the corresponding labels.
        :return: None
        """
        self.model.fit(X_train, y_train)


    def predict(self, X_test):
        """
        Generates predictions using the trained Random Forest model.

        :param X_test: A NumPy array of shape (num_samples, num_features) containing the test images.
        :return: A NumPy array of predicted class labels (digits 0-9).
        """
        return self.model.predict(X_test)


