# **MNIST Classifier**

This project implements a basic neural network to classify handwritten digits from the MNIST dataset using PyTorch. The model is designed to achieve at least 90% accuracy on the test dataset and provides visualizations of training progress and classification results.

---

## **Table of Contents**
1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)

---

## **Features**
- Load and preprocess the MNIST dataset.
- Train a fully connected neural network on the dataset.
- Evaluate the model's performance on the test dataset.
- Visualize training losses and test accuracies.
- Display sample predictions with corresponding labels.

---

## **Installation**

Follow these steps to set up and run the project:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/mnist-classifier.git
   cd mnist-classifier
2. **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
4. **Download the MNIST dataset**: The dataset will automatically be downloaded during training.

---

## **Usage**

1. **Train/Test The Model. Scripts**:

- python -m scripts.train_model
- python -m scripts.test_model
- python -m scripts.test_visualize_model
- python -m scripts.train_test_model


