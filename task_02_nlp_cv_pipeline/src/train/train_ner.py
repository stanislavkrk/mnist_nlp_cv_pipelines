import json
import os
from task_02_nlp_cv_pipeline.models.universal_train_inference import UniversalTrainInference


def train_ner() -> None:
    """
    Trains the Named Entity Recognition (NER) model using a preprocessed dataset.

    This function loads training data from a specified JSON file, extracts tokens and labels,
    and trains an NER model if pre-trained weights are not available.

    :raises FileNotFoundError: If the dataset file does not exist.
    :return: None
    """
    # Define the dataset path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "../../data/ner/generated_animal_dataset.json")

    # Ensure the dataset file exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"File  {dataset_path} isn`t exist.")

    # Load dataset from JSON file
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    # Extract training data: tokens and labels
    X_train = [sample["tokens"] for sample in dataset["train"]]
    y_train = [sample["labels"] for sample in dataset["train"]]

    # Define the model weights path
    current_dir02 = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(current_dir02, "../../weights/ner_model_weights.h5")

    # Initialize the NER model trainer
    trainer = UniversalTrainInference(model_type="ner")

    # Train the model if weights do not exist
    if not os.path.exists(weights_path):
        print("Weights aren`t exist! Starting learning.")
        trainer.train_universal(X_train, y_train)
    else:
        print("Weights are exist! Skip learning.")

