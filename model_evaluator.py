# Import required libraries for loading preprocessed data
import joblib
import numpy as np
from scipy import sparse

# Load the TF-IDF vectorized feature matrices from .npz files
X_train = sparse.load_npz("X_train.npz")
X_test = sparse.load_npz("X_test.npz")

# Load the encoded sentiment labels from .pkl files
y_train = joblib.load("y_train.pkl")
y_test = joblib.load("y_test.pkl")

print("Files loaded successfully.")
print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

# Import the logistic regression model and evaluation metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize and train the logistic regression model
print("\nTraining the model...")
model = LogisticRegression(max_iter=1000)  # Increased max_iter to ensure convergence
model.fit(X_train, y_train)

# Predict the labels on the test set
print("Making predictions...")
y_pred = model.predict(X_test)

# Evaluate the model using accuracy and classification metrics
accuracy = accuracy_score(y_test, y_pred)
print("\nEvaluation Results:")
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save predictions to a CSV file for visualization or further analysis
import pandas as pd
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results_df.to_csv("model_predictions.csv", index=False)
print("Saved: model_predictions.csv")

# Save classification metrics to a text file
with open("metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred))
print("Saved: metrics.txt")

# Generate and save the confusion matrix as a PNG image
import matplotlib.pyplot as plt
import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
print("Saved: confusion_matrix.png")
