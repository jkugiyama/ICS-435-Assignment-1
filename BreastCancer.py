import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ----- Load Dataset -----
data = load_breast_cancer()
X = data.data
y = data.target

# ----- Train-Test Split (80/20) -----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----- Feature Scaling (for KNN) -----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Helper function for evaluation
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred)
    }

    cm = confusion_matrix(y_test, y_pred)

    # Create a new figure for each model
    plt.figure()
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=data.target_names
    )
    disp.plot()
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()

    return metrics

results = []

# ----- KNN Model -----
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
results.append(evaluate_model(knn, X_test_scaled, y_test, "KNN (k=5)"))

# ----- Decision Tree Model -----
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
results.append(evaluate_model(dt, X_test, y_test, "Decision Tree (default)"))

# ----- Random Forest Model -----
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
results.append(evaluate_model(rf, X_test, y_test, "Random Forest (100 trees)"))

# ----- Results  -----
results_df = pd.DataFrame(results)
print(results_df)

# ----- Ablation Study -----
print("\nAblation Study Results\n")

# KNN: different k values
for k in [3, 7, 11]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    print(f"KNN (k={k}) Accuracy:", accuracy_score(y_test, y_pred))

# Decision Tree: different max_depth
for depth in [3, 5, 10]:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    print(f"Decision Tree (max_depth={depth}) Accuracy:", accuracy_score(y_test, y_pred))

# Random Forest: different max_depth
for depth in [3, 5, None]:
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=depth, random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print(f"Random Forest (max_depth={depth}) Accuracy:", accuracy_score(y_test, y_pred))
