from task_02_nlp_cv_pipeline.models.universal_train_inference import UniversalTrainInference

def inference_img(image_path: str) -> str:
    """
    Runs inference on an image using the pre-trained image classification model.

    :param image_path: Path to the image file that needs to be classified.
    :return: Predicted class label for the image.
    """
    # Initialize the image classification model
    model = UniversalTrainInference(model_type="img")

    # Set the image path for inference
    model.item = image_path

    # Perform inference and return the predicted class
    return model.inference_universal()