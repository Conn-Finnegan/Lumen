# src/evaluate.py

from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from load_data import load_data


def evaluate_model(model_path, data_dir):
    # Load the trained model
    model = load_model(model_path)

    # Load the validation data only
    _, X_val, _, y_val = load_data(data_dir=data_dir)

    # Make predictions (sigmoid outputs between 0 and 1)
    y_pred_probs = model.predict(X_val)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    # Print classification report
    print("\n📊 Classification Report:")
    print(
        classification_report(
            y_val, y_pred, target_names=["Non-cancerous", "Cancerous"]
        )
    )

    # Plot confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-cancerous", "Cancerous"],
        yticklabels=["Non-cancerous", "Cancerous"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(project_root, "models/cancer_classifier.h5")  # or .keras
    data_path = os.path.join(project_root, "data")
    evaluate_model(model_path, data_path)
