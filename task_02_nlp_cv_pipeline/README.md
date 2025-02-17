# 🦡 NLP & CV Pipeline

## This project combines **Natural Language Processing (NLP)** and **Computer Vision (CV)** to verify whether an image contains the animal mentioned in the given text.

---

## Approach and Model Details

This solution is built using a **dual-model pipeline**, where both textual and visual data are processed to validate the user's input:

### Named Entity Recognition (NER) Model (NLP)

- The **NLP component** is responsible for **extracting animal names** from the text input.
- We utilize a transformer-based model **distilBERT-base-uncased**, fine-tuned on domain-specific data to improve entity recognition accuracy.
- The model is trained using **cross-entropy loss**, optimized with **Adam**, and regularized using **dropout**.
- The output of this model is a **structured entity list**, identifying whether an animal is mentioned.

### Image Classification Model (CV)

- This component is responsible for **identifying which animal is present in the image**.
- We choose a **Convolutional Neural Network (CNN)** architecture, specifically **MobileNetV2**, pre-trained on large-scale image datasets (such as **ImageNet**) and fine-tuned on our curated dataset.
- The loss function is **categorical cross-entropy**, with an **Adam optimizer** and **learning rate scheduling** for efficient convergence.
- The final model **classifies an image into one of 10+ animal categories**.

---

### End-to-End Pipeline Flow

- **Input** → The user provides a **text description** (e.g., _"There is a cow in the picture."_) along with an **image**.  
- **NER Model** → Extracts the mentioned animal from the text.  
- **Image Classifier** → Predicts which animal is present in the image.  
- **Verification Logic** → Compares both results to return a **Boolean output** (`True` if the animal in text and image matches, `False` otherwise).

---

This hybrid approach ensures a **multi-modal verification system**, combining the strengths of both **NLP and CV** models to achieve accurate results.

## Project Pipeline

This pipeline consists of:
1. **Named Entity Recognition (NER) model** – extracts animal names from text.
2. **Image Classification model** – identifies the animal in the image.
3. **Pipeline script** – takes a text and an image as input and outputs `True` if the image and text contains the mentioned animal, otherwise `False`.

## 📂 Project Structure

```
task_02_nlp_cv_pipeline/
│── data/
│   ├── img/
│   │   ├── image_for_recognize/
│   │   ├── images_preprocessed/
│   ├── ner/
│── models/
│── src/
│── weights/
│── ner_img_classification_jupyter.ipynb
│── README.md
│── requirements.txt
│── .gitignore
```
---

## 🛠 Dataset of Animals Images Installation

### Training Models
All models are already trained and weight files are included in the project.
Weights for **Image-Classificator** is already in folder **/weights** and **ready to inference**.
#### But file of NER model weights is too large and you can download it by link [->NER weights](https://drive.google.com/file/d/1YPFIIulgqk5fMbNfPS7if_IBdN0UBWWq/view?usp=sharing)
You need download it into **/weights** folder.

So, **before training IMG-Classifier model** you need to **unpack ZIP archive** of dataset.

To avoid excessive small file commits, the dataset is provided as a **ZIP archive**.  
Before running the project, **extract the dataset** as follows:

### **Manual Extraction**
1. Navigate to the dataset directory:
   ```sh
   cd task_02_nlp_cv_pipeline/data/img/images_preprocessed/
   ```
2. Unzip here the dataset:
   ```sh
   unzip all_animals_extract_here_as_different_folders.zip
   ```
3. After extraction, the folder structure should look like:
   ```
   images_preprocessed/
   ├-- butterfly/
   ├-- cat/
   ├-- chicken/
   ├-- cow/
   ├-- ...
   └-- zebra/
   ```

---

## ⚙️ Environment Setup

### **Create a Virtual Environment**
```
python -m venv .venv
source venv/bin/activate   # On macOS/Linux
.venv\Scripts\activate      # On Windows
```

### **Install Dependencies**
```
pip install -r requirements.txt
```

---

## 🎬 How to Run the Project

The project provides two main files for execution and analysis:

### **Running the Main Pipeline** (`src/pipeline.py`)
This script is the **core execution pipeline**. It follows the logic:
- **If models are not trained** (i.e., weight files are missing), it will **automatically train the models**.
- **If models are already trained** (default case), it will **directly proceed with predictions**.
- **No additional setup is required**, as all necessary data for prediction (text and images) is already prepared.

### Training Models
All models are already trained and weight files are included in the project.
Weights for **Image-Classificator** is already in folder **/weights** and **ready to inference**.
#### But file of NER model weights is too large and you can download it by link [->NER weights](https://drive.google.com/file/d/1YPFIIulgqk5fMbNfPS7if_IBdN0UBWWq/view?usp=sharing)
You need download it into **/weights** folder.

If you want to **train the model from scratch**, simply delete, move to another folder, or rename the **existing weight** files located in the **/weights** directory.


### Exploring the Jupyter Notebook (`ner_img_classification_jupyter.ipynb`)

This Jupyter Notebook provides a detailed overview of:

- **Dataset structure** and **preprocessing**
- **Selected models** and their **architectures**
- **Training logic** and **methodology** 
- **Analysis of results** and **edge cases**

---

