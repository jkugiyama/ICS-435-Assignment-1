# Machine Learning Models Evaluation
## Project Overview
This assignment compares the performance of three machine learning classifiers: K-Nearest Neighbors (KNN), Decision Tree, Random Forest
The models are trained and evaluated on the Breast Cancer dataset from sklearn. The goal is to classify tumors as malignant or benign and compare model performance using multiple evaluation metrics.

## Dataset
Dataset: sklearn.datasets.load_breast_cancer()

Samples: 569

Features: 30 numerical features

Task: Binary classification (malignant vs. benign)

The dataset is split into 80% training data and 20% testing data


## Preprocessing
StandardScaler is applied for KNN because it relies on distance calculations. Decision Tree and Random Forest are trained on unscaled data.


## Models Implemented

K-Nearest Neighbors (k=5)

Decision Tree (default parameters + depth tuning)

Random Forest (100 trees + depth tuning)


## Evaluation Metrics

Each model is evaluated using: Accuracy, Precision, Recall, F1-score, Confusion Matrix


## Ablation Study

Hyperparameters tested:

- KNN: Different values of k (3, 7, 11)

- Decision Tree: Different max_depth values (3, 5, 10)

- Random Forest: Different max_depth values (3, 5, None)


## How to Run

Make sure required packages are installed:
```
pip install numpy pandas matplotlib scikit-learn
```

Then run:
```
python BreastCancer.py
```

Confusion matrices will display and performance results will print in the terminal.
