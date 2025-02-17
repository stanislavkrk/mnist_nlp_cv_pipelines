from train.train_ner import train_ner
from train.train_image_classifier import train_img
from inference.inference_ner import inference_ner
from inference.inference_image_classifier import inference_img
import os

def normalize_nlp_output(nlp_prediction: set) -> str:
    """
    Normalizes the NER model's output by extracting and formatting the predicted class name.

    :param nlp_prediction: A set containing the predicted named entity.
    :return: Normalized class name as a lowercase string.
    """
    # Extract the first predicted class and format it
    class_name = list(nlp_prediction)[0]
    class_name = class_name.split("-")[-1].lower()

    return class_name

def pipeline(text: str, image_path: str) -> bool:
    """
    Executes the AI pipeline to verify if the mentioned animal in the text matches the one in the image.

    - Ensures the required model weights exist.
    - Runs NER inference on the text to extract the animal name.
    - Runs Image Classification inference on the provided image.
    - Compares the results from both models.

    :param text: The input text containing an animal reference.
    :param image_path: The file path to the image for classification.
    :return: True if the detected animal in the text matches the one in the image, otherwise False.
    """
    print("Checking NER weight file")
    train_ner()

    print("Checking IMG weight file")
    train_img()

    print('Running NER-inference')
    predicted_text_animal = inference_ner(text)

    print('Running Image-Classification inference')
    predicted_image_animal = inference_img(image_path)

    # Compare the normalized NER output with the image classification result
    return  normalize_nlp_output(predicted_text_animal) == predicted_image_animal

if __name__ == "__main__":

    # Define the text input
    text_input = "Who saw the elephant?"

    # Define the image file path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, "../data/img/image_for_recognize/Elephant_12.jpg")

    # Run the AI pipeline
    result = pipeline(text_input, image_path)

    print(result)
