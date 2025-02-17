import tensorflow as tf
from transformers import TFAutoModelForTokenClassification, AutoTokenizer
from .default_interface import ModelInterface
import numpy as np
from collections import Counter

class NerModel(ModelInterface):
    """
    Named Entity Recognition (NER) model for detecting animal names in text.

    This class implements a transformer-based NER model, using a pre-trained
    language model with fine-tuning. It provides methods for encoding data,
    training, predicting named entities, and saving/loading model weights.
    """

    def __init__(self, model_name:str = "distilbert-base-uncased"):
        """
        Initializes the NER model with a pre-trained transformer.

        :param model_name: The name of the Hugging Face model to load.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load base model with 11 classes (10 animals + "O" for non-entity tokens)
        self.model = TFAutoModelForTokenClassification.from_pretrained(model_name, num_labels=11)

        # Freeze all layers except the final classification layer (fine-tuning)
        for layer in self.model.layers[:-1]:
            layer.trainable = False

        # Lable-map fo animals
        self.label_map = {
            "O": 0,
            "B-BUTTERFLY": 1, "B-CAT": 2, "B-CHICKEN": 3, "B-COW": 4, "B-DOG": 5,
            "B-ELEPHANT": 6, "B-HORSE": 7, "B-SHEEP": 8, "B-SQUIRREL": 9, "B-ZEBRA": 10
        }

    def encode_data(self, texts: list, labels: list) -> tuple:
        """
        Tokenizes input texts and aligns labels with tokenized words.

        :param texts: A list of tokenized sentences (word-level).
        :param labels: A list of corresponding labels for each token in the sentence.
        :return: A tuple containing tokenized input tensors and encoded label arrays.
        """
        # Tokenize entire sentences while keeping word alignment
        tokenized_inputs = self.tokenizer(
            [" ".join(tokens) for tokens in texts],
            padding=True, truncation=True, return_tensors="tf", max_length=128
        )
        # Map subwords to original word indices
        encoded_labels = []
        for i, label in enumerate(labels):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []

            previous_word_id = None
            for word_id in word_ids:
                if word_id is None or word_id >= len(label):
                    label_ids.append(-100) # Ignore padding and special tokens
                elif word_id != previous_word_id:
                    # Assign the first subword of a token its original label
                    filtered_label = self.label_map.get(label[word_id], 0)
                    label_ids.append(filtered_label)
                else:
                    # Ignore subsequent subwords of a token
                    label_ids.append(-100)

                previous_word_id = word_id

            encoded_labels.append(label_ids)

        return tokenized_inputs, np.array(encoded_labels)

    def train(self, X_train, y_train, epochs:int = 5, batch_size:int = 8):
        print("Train data tokenization...")
        train_inputs, train_labels = self.encode_data(X_train, y_train)

        # Unic-labels print for checking
        unique_labels, counts = np.unique(train_labels, return_counts=True)
        print(f"Unic labels in data-set: {dict(zip(unique_labels, counts))}")

        # If all labels are zero, the map is failed
        if len(unique_labels) == 1 and unique_labels[0] == 0:
            raise ValueError("All labels are 'O'! Take revision of map!")

        # Delete -100 (ignored subwords)
        train_labels = np.where(train_labels == -100, 0, train_labels)

        # Class balancing (loss weight for "O", more weigth for rare classes)
        label_counts = Counter(train_labels.flatten())
        total_count = sum(label_counts.values())
        class_weights = {label: total_count / (len(label_counts) * count) for label, count in label_counts.items()}

        # Making sample_weights with correct dimensions (batch_size, sequence_length)
        sample_weights = np.vectorize(class_weights.get)(train_labels)

        # Making Dataset with sample_weights
        train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_inputs), train_labels, sample_weights)).shuffle(1000).batch(batch_size)

        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # Adding weighted loss for compilation
        self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

        print("Start fine-tuning NER model...")
        self.model.fit(train_dataset, epochs=epochs)

    def predict(self, text: str) -> set:
        """
        Predicts named entities in a given text.

        :param text: The input text containing possible animal names.
        :return: A set of detected named entities.
        """
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="tf")

        # Get model predictions (logits) and extract class probabilities
        outputs = self.model(**inputs).logits
        predictions = tf.argmax(outputs, axis=-1).numpy()

        # Convert input token indices back to words
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].numpy()[0])

        predicted_labels = []
        self.found_labels = set()  # Clear th list of founded labels before new prediction

        for token, label in zip(tokens, predictions[0]):
            predicted_label = list(self.label_map.keys())[label]
            predicted_labels.append((token, predicted_label))

            # Store only animal-related entities (ignoring "O" class)
            if predicted_label != "O":  # For animal labels only
                self.found_labels.add(predicted_label)  # Add to list of founded labels
        print("Text:", predicted_labels)
        return self.found_labels

    def save_weights(self, path: str):
        """
        Saves the model weights to a file.

        :param path: Path to save the weights.
        :return: None
        """
        self.model.save_weights(path)

    def load_weights(self, path: str):
        """
        Loads the model weights from a file.

        :param path: Path to load the weights from.
        :return: None
        """
        self.model.load_weights(path)
