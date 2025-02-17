from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import json
import os

from .default_interface import ModelInterface

class ImageClassifier(ModelInterface):
    """
    Image classification model based on MobileNetV2.

    This class implements an image classifier using a pre-trained MobileNetV2 model
    with additional fully connected layers. It extends the ModelInterface and
    provides methods for training, prediction, and weight management.
    """

    def __init__(self, num_classes:int = 10):
        """
        Initializes the ImageClassifier with a modified MobileNetV2 architecture.

        :param num_classes: Number of target classes for classification (default is 10).
        """
        # Use MobileNetV2 with pretrained weights
        base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

        # Add custom classification layers on top of the base model
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.5)(x)
        output_layer = Dense(num_classes, activation="softmax")(x)

        # Create the final model
        self.model = Model(inputs=base_model.input, outputs=output_layer)

        # Compile the model with Adam optimizer and sparse categorical cross-entropy loss
        self.model.compile(optimizer=Adam(learning_rate=0.0001),
                           loss="sparse_categorical_crossentropy",
                           metrics=["accuracy"])

        # Load class mapping from a JSON file if available, otherwise create a default mapping
        current_dir = os.path.dirname(os.path.abspath(__file__))
        class_map_path = os.path.join(current_dir, "../data/img/images_preprocessed/class_map.json")
        if os.path.exists(class_map_path):
            with open(class_map_path, "r") as f:
                self.class_map = json.load(f)
        else:
            self.class_map = {str(i): f"class_{i}" for i in range(num_classes)}

    def train(self, X_train, y_train, epochs:int = 5, batch_size:int = 16):
        """
        Trains the model on the provided dataset.

        :param X_train: Training images in an appropriate format (e.g., NumPy array).
        :param y_train: Labels for the training images.
        :param epochs: Number of training iterations over the dataset (default is 5).
        :param batch_size: Number of samples processed per training step (default is 16).
        :return: None
        """
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, image_path: str) -> str:
        """
        Predicts the class label for the given image.

        :param image_path: Path to the image file.
        :return: Predicted class label as a string.
        """
        # Load and preprocess the input image
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Perform prediction
        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions)

        # Return the class label
        return self.class_map.get(str(predicted_class))

    def save_weights(self, path: str):
        """
        Saves the trained model weights to a specified file.

        :param path: File path where the model weights should be saved.
        :return: None
        """
        self.model.save_weights(path)

    def load_weights(self, path: str):
        """
        Loads the model weights from a specified file.

        :param path: File path from which the model weights should be loaded.
        :return: None
        """
        self.model.load_weights(path)
