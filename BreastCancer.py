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
X = data.data     # Feature matrix (input variables)
y = data.target   # Target labels (0 = malignant, 1 = benign)

# ----- Train-Test Split -----
# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----- Feature Scaling (for KNN) -----
# Since KNN is sensitive to feature scales, we standardize the features
# Decision Tree and Random Forest do not require feature scaling, 
# but we will use the same scaled features for consistency in evaluation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit scaler only on training data
X_test_scaled = scaler.transform(X_test)        # Apply same scaling to test data

# Helper function for evaluation
# Predict and calculate metrics, and plot confusion matrix
# Includes a parameter for the axis to plot on, 
# allowing us to create a single figure with multiple confusion matrices
def evaluate_model(model, X_test, y_test, model_name, ax):
    y_pred = model.predict(X_test) # Generate predictions on the test set

    # Store evaluation metrics in a dictionary
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred)
    }

    cm = confusion_matrix(y_test, y_pred)  # Compute confusion matrix

    # Create confusion matrix display
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=data.target_names
    )
    # Plot the confusion matrix on the provided axis
    disp.plot(ax=ax)
    ax.set_title(f"{model_name} Confusion Matrix")

    return metrics  # Return performance metrics for this model

# Create a single figure for all models
# 1 row and 3 columns of subplots for the three models
fig, axs = plt.subplots(1, 3, figsize=(15, 5))  
results = []   # List to store results for each model 

# ----- KNN Model -----
knn = KNeighborsClassifier(n_neighbors=5)  # Initialize KNN with number of neighbors = 5
knn.fit(X_train_scaled, y_train)           # Train on scaled training data
# Evaluate KNN and store results
results.append(evaluate_model(knn, X_test_scaled, y_test, "KNN (k=5)", axs[0]))

# ----- Decision Tree Model -----
dt = DecisionTreeClassifier(random_state=42)  # Default decision tree
dt.fit(X_train, y_train)                      # Train on original unscaled training data 
# Evaluate Decision Tree and store results
results.append(evaluate_model(dt, X_test, y_test, "Decision Tree (default)", axs[1]))

# ----- Random Forest Model -----
rf = RandomForestClassifier(n_estimators=100, random_state=42) # Initialize Random Forest with 100 trees
rf.fit(X_train, y_train)                                # Train on original unscaled training data
# Evaluate Random Forest and store results
results.append(evaluate_model(rf, X_test, y_test, "Random Forest (100 trees)", axs[2]))

# Show the figure with all confusion matrices
plt.tight_layout()
plt.show()

# ----- Results  -----
results_df = pd.DataFrame(results)    # Convert results list to a pandas DataFrame for easier visualization
print(results_df)                     # Print the performance metrics for all models in a table

# ----- Ablation Study -----
# Test how different hyperparameters affect model performance
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
