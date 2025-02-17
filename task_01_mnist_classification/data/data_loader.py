from tensorflow.keras.datasets import mnist

class MNISTDataLoader:
    """
    Data loader for the MNIST dataset.

    This class loads the MNIST dataset and provides methods to retrieve training and test data.
    The dataset consists of grayscale images of handwritten digits (0-9) with a resolution of 28x28 pixels.
    """

    def __init__(self):
        """
        Initializes the MNIST data loader by loading the dataset into memory.

        The dataset is loaded into four NumPy arrays:
        - X_train: Training images (shape: [60000, 28, 28]).
        - y_train: Training labels (shape: [60000]).
        - X_test: Test images (shape: [10000, 28, 28]).
        - y_test: Test labels (shape: [10000]).
        """
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()

    def get_train_data(self):
        """
        Returns the training data.

        :return: A tuple (X_train, y_train) where:
                 - X_train is a NumPy array of shape (60000, 28, 28) containing training images.
                 - y_train is a NumPy array of shape (60000,) containing corresponding labels.
        """
        return self.X_train, self.y_train

    def get_test_data(self):
        """
        Retrieves the test data.

        :return: A tuple (X_test, y_test) where:
                 - X_test is a NumPy array of shape (10000, 28, 28) containing test images.
                 - y_test is a NumPy array of shape (10000,) containing corresponding labels.
        """
        return self.X_test, self.y_test