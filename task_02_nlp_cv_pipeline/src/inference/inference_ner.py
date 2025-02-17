from task_02_nlp_cv_pipeline.models.universal_train_inference import UniversalTrainInference

def inference_ner(text: str):
    """
    Runs inference on a given text using the pre-trained Named Entity Recognition (NER) model.

    :param text: The input text containing possible named entities (e.g., animal names).
    :return: A set of predicted named entities found in the text.
    """
    # Initialize the NER model
    model = UniversalTrainInference(model_type="ner")

    # Set the input text for inference
    model.item = text

    # Perform inference and return the predicted named entities as a set
    return model.inference_universal()
