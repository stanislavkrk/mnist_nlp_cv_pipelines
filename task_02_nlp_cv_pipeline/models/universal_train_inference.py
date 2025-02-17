from .model_ner import NerModel
from.model_image_classifier import ImageClassifier
import os

class UniversalTrainInference:
    """
    A universal class for training and inference of different models.

    This class supports two types of models:
    - Named Entity Recognition (NER) using DistilBERT
    - Image Classification using MobileNetV2

    Depending on the chosen model type, it initializes the corresponding model,
    assigns appropriate weight file paths, and defines input samples for inference.
    """

    def __init__(self, model_type:str = 'default'):
        """
        Initializes the class and selects the appropriate model.

        :param model_type: The type of model to use ('ner' for NLP, 'img' for image classification).
        :raises ValueError: If an unsupported model type is provided.
        """
        self.current_dir = os.path.dirname(os.path.abspath(__file__))

        # Match the model type to initialize the correct model
        match model_type:
            case "ner":
                self.model = NerModel()
                self.weights_file_name = os.path.abspath(os.path.join(self.current_dir, "../weights/ner_model_weights.h5"))
                self.model_name = 'DistilBERT-Base-Uncased'
                self.item = "There is a cat"
                self.message = "Predicted NLP class:"
            case "img":
                self.model = ImageClassifier()
                self.weights_file_name = os.path.abspath(os.path.join(self.current_dir, "../weights/img_model_weights.h5"))
                self.model_name = 'MobileNetV2'
                self.item = "data/images/cat.jpg"
                self.message = "Predicted IMG class:"
            case _:
                raise ValueError(f"Unknown model type: {model_type}. Use 'ner' or 'img'.")

    def train_universal(self, X_train, y_train):
        """
        Trains the selected model and saves its weights.

        :param X_train: Training data (text for NER, images for classification).
        :param y_train: Corresponding labels (NER tags or image class labels).
        :return: None
        """
        self.model.train(X_train, y_train)

        # Ensure the weights directory exists before saving
        weights_dir = os.path.dirname(self.weights_file_name)
        os.makedirs(weights_dir, exist_ok=True)

        # Save trained weights
        self.model.save_weights(self.weights_file_name)
        print((f"Weights saved for this model: {self.model_name}"))

    def inference_universal(self) -> str:
        """
        Loads the trained model weights and runs inference on a sample input.

        :return: The predicted output from the model.
        """
        # Load pre-trained model weights
        self.model.load_weights(self.weights_file_name)

        # Perform inference on the predefined sample item
        prediction = self.model.predict(self.item)

        # Print and return the prediction result
        print(f"{self.message} {prediction}")
        return prediction