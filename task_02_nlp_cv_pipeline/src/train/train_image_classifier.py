from task_02_nlp_cv_pipeline.models.universal_train_inference import UniversalTrainInference
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

def train_img() -> None:
    """
    Trains the image classification model using preprocessed image data.

    This function loads training images from a specified directory, applies preprocessing,
    and trains a deep learning model if pre-trained weights are not available.

    :raises FileNotFoundError: If the class map file does not exist.
    :return: None
    """
    # Define directories and constants
    current_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(current_dir, "../../data/img/images_preprocessed")
    BATCH_SIZE = 32
    IMG_SIZE = (224, 224)

    # Load class map from json
    class_map_path = os.path.join(DATA_DIR, "class_map.json")
    if os.path.exists(class_map_path):
        with open(class_map_path, "r") as f:
            class_map = json.load(f)
    else:
        raise FileNotFoundError(f"Doesn`t exist {class_map_path}")

    # Initialize data generator for training
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
    )

    # Load training data from directory
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        subset="training"
    )

    # Collect training data from batches
    X_train, y_train = [], []
    for _ in range(len(train_generator)):
        batch_x, batch_y = next(train_generator)
        X_train.append(batch_x)
        y_train.append(batch_y)

    # Convert lists to NumPy arrays
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    # Define path for model weights
    current_dir02 = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(current_dir02, "../../weights/img_model_weights.h5")

    # Initialize the image classification trainer
    trainer = UniversalTrainInference(model_type="img")

    # Train the model if weights are not available
    if not os.path.exists(weights_path):
        print("Weights aren`t exist! Starting learning.")
        trainer.train_universal(X_train, y_train)
    else:
        print("Weights are exist! Skip learning.")


if __name__ == "__main__":
    train_img()