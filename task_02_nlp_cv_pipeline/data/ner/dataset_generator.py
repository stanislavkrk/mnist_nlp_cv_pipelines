import json
import random
import os

ANIMALS = ["butterfly", "cat", "chicken", "cow", "dog",
           "elephant", "horse", "sheep", "squirrel", "zebra"]

TEMPLATES = [
    "I saw a {} in the park.",
    "A {} was running across the road.",
    "Have you ever seen a {} up close?",
    "The zoo has a new {} exhibit.",
    "My friend has a pet {} at home.",
    "There is a {} in the field.",
    "A {} appeared in my backyard this morning.",
    "The {} is known for its unique behavior.",
    "Scientists are studying the {} in the wild.",
    "A group of {}s was spotted near the river.",
    "Farmers often keep {}s for their needs.",
    "A {} was found hiding under the tree.",
    "Hunters rarely see {}s in this region.",
    "The movie featured a trained {} in several scenes.",
    "Children love playing with {} toys.",
    "A photographer captured an amazing shot of a {}.",
    "The {} ran swiftly through the grass.",
    "People used to believe {}s were mythical creatures.",
    "A {} was rescued by animal control yesterday.",
    "The {} is commonly found in forests.",
    "Local folklore includes stories about {}s.",
    "The {} made a strange noise last night.",
    "Zookeepers take special care of {}s.",
    "A {} accidentally wandered into the city streets.",
    "The {} is an essential part of the ecosystem.",
    "A wounded {} was found near the lake.",
    "Bird watchers were excited to see a {}.",
    "A {} is often used as a symbol of strength.",
    "The circus once had a {} as part of the show.",
    "Legend says a {} saved a village long ago.",
    "A {} was spotted near the mountains last week.",
    "Farmers raise {}s for their wool and meat.",
    "Many fairy tales include a friendly {}.",
    "A {} was seen playing with other animals.",
    "A {} is faster than most people think.",
    "Old paintings often depict a {} in nature.",
    "A {} can be very friendly if raised properly.",
    "The forest is home to many {}s.",
    "Some people keep {}s as exotic pets.",
    "A baby {} was recently born in the zoo.",
    "The {} was staring at me through the fence.",
    "Have you ever fed a {}?",
    "The sound of a {} can be heard from afar.",
    "A {} was resting under the shade.",
    "A {} is an interesting creature to study.",
    "The {} looked very hungry.",
    "A {} was running alongside our car.",
    "The {} was standing near the lake.",
    "A {} escaped from the farm!",
    "A {} jumped over the fence.",
    "The {} was carefully observing its surroundings.",
    "A {} was hiding in the tall grass."
]


def generate_annotated_sentences(num_samples=3000):
    """
    Generates a dataset of annotated sentences for Named Entity Recognition (NER).

    Each generated sentence contains an animal name, with its corresponding NER label.
    The dataset is split into training, validation, and test sets.

    :param num_samples: Number of sentences to generate.
    :return: A dictionary containing 'train', 'validation', and 'test' datasets.
    """
    dataset = {"train": [], "validation": [], "test": []}

    for _ in range(num_samples):
        animal = random.choice(ANIMALS) # Randomly select an animal
        sentence = random.choice(TEMPLATES).format(animal) # Fill template with an animal name
        tokens = sentence.split() # Tokenize sentence

        # Assign "O" (Outside) to all tokens by default
        labels = ["O"] * len(tokens)

        # Assign B-{ENTITY} label to the animal name in the sentence
        for i, token in enumerate(tokens):
            if token.lower() == animal:
                labels[i] = f"B-{animal.upper()}"

        # Randomly assign the sentence to train, validation, or test set
        dataset_type = random.choices(["train", "validation", "test"], weights=[0.7, 0.2, 0.1])[0]
        dataset[dataset_type].append({"tokens": tokens, "labels": labels})

    return dataset


def save_dataset(dataset: dict, output_path: str) -> None:
    """
    Saves the generated dataset to a JSON file.

    :param dataset: The dataset dictionary containing 'train', 'validation', and 'test' splits.
    :param output_path: The file path to save the JSON dataset.
    :return: None
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=4)

    print(f"Generated dataset saved to {output_path}")


if __name__ == "__main__":
    # Generate and save dataset
    dataset_size = 5000
    generated_dataset = generate_annotated_sentences(dataset_size)

    output_path = "./generated_animal_dataset.json"
    save_dataset(generated_dataset, output_path)
