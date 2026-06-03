# Digit Recognizer (MNIST) 🔢

## 📌 Project Overview
The objective of this project is to classify grayscale images of handwritten digits (0-9) into their respective categories. This serves as a foundational multiclass image classification challenge using the well-known MNIST dataset. The primary focus is on establishing a strong baseline model and conducting in-depth error analysis using confusion matrices.

## 🛠 Tech Stack
* **Language:** Python
* **Libraries:** `scikit-learn`, `pandas`, `numpy`, `matplotlib`

## 🧠 Methodology & Architecture

### 1. Data Preparation & Visualization
The dataset consists of 784 pixel features representing 28x28 pixel images. Initial exploration included reshaping the 1D pixel arrays back into 2D matrices to visually verify the digit representations. The training data was split using an 80/20 ratio for local validation.

### 2. Baseline Modeling
A Stochastic Gradient Descent (`SGDClassifier`) was selected as the baseline model for this multiclass classification task. To ensure model stability and reliable performance estimates before evaluating the test set, 3-fold cross-validation (`cross_val_score`) was utilized.

### 3. Error Analysis
A significant portion of the analysis was dedicated to understanding the model's mistakes:
* Computed a Confusion Matrix using `cross_val_predict`.
* Normalized the matrix by row sums and zeroed out the diagonal to visually isolate and highlight specific misclassifications.
* Identified that the model most frequently confused the digit '5' with the digit '8' (286 occurrences).
* Extracted and plotted these specific false predictions to visually analyze why the model failed on those particular handwriting styles.

## 📊 Results
* **Local Cross-Validation Mean Accuracy:** ~87.4%
* **Final Submission:** The model was trained on the provided training set and predictions were mapped to `ImageId`s for the final Kaggle `submission.csv`.

## 🚀 Local Setup
To replicate the environment and run the notebook locally:

1. Clone the repository to your local machine.
2. Install the required dependencies (using `uv` or `pip`):
```bash
pip install pandas numpy matplotlib scikit-learn
