# MNIST Classification

This project implements **MNIST digit classification** using different machine learning models, all of which are analyzed and executed within a Jupyter Notebook.

The classification task is performed using three different models:

- **Random Forest (RF)** – A traditional machine learning model used as a baseline.
- **Feed-Forward Neural Network (FFNN)** – A fully connected deep learning model.
- **Convolutional Neural Network (CNN)** – A model optimized for image classification.

To ensure a structured and maintainable approach, the project follows **Object-Oriented Programming (OOP)** principles.  
A **general abstract interface** is implemented, which standardizes training and inference across different model architectures.

---

## Project Pipeline

The project consists of the following components:
1. **Data Processing** – Scripts for loading and preprocessing the MNIST dataset.
2. **Model Implementations** – Different classification models for digit recognition.
3. **Jupyter Notebook** – The main execution file, containing all logic for data handling, model training, evaluation, and visualization.

---

## 📂 Project Structure

```
task_01_mnist_classification/
│── data/                            # Contains scripts for data loading and preprocessing
│── models/                          # Contains model implementations (not used directly)
│── mnist_pipeline_jupyter.ipynb     # Main execution file with full pipeline
│── README.md                        # This documentation
│── requirements.txt                  # Dependencies for the project
│── .gitignore
```

---

## ⚙️ Environment Setup

### **Create a Virtual Environment**
```
python -m venv .venv
source .venv/bin/activate   # On macOS/Linux
.venv\Scripts\activate      # On Windows
```

### **Install Dependencies**
```
pip install -r requirements.txt
```

---

## 🎬 How to Run the Project

The entire classification pipeline, including dataset loading, preprocessing, model training, evaluation, and visualization, is implemented in a **Jupyter Notebook**.

### ▶ **Running the Notebook** (`mnist_pipeline_jupyter.ipynb`)
To open and execute the notebook, run:
```
jupyter notebook mnist_pipeline_jupyter.ipynb
```

Inside the notebook, you will find:
- **Dataset analysis and preprocessing insights**
- **Model architectures and training logic**
- **Evaluation metrics and performance comparisons**
- **Visualization of predictions and edge cases**

---

