import os
from datetime import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from load_data import load_data


def evaluate_model(model_path, data_dir):
    # Load model
    model = load_model(model_path)

    # Load validation data
    _, X_val, _, y_val = load_data(data_dir=data_dir)

    # Predict
    y_pred_probs = model.predict(X_val, verbose=1)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    # Print classification report
    print("\n📊 Classification Report:")
    print(
        classification_report(
            y_val, y_pred, target_names=["Non-cancerous", "Cancerous"]
        )
    )

    # Generate and plot confusion matrix
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

    # Ensure output folder exists
    os.makedirs("outputs", exist_ok=True)

    # Save confusion matrix with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    plot_path = f"outputs/confusion_matrix_{timestamp}.png"
    plt.savefig(plot_path)
    plt.show()

    print(f"📷 Confusion matrix saved to: {plot_path}")


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(__file__))

    # Automatically find the latest .keras model
    model_dir = os.path.join(project_root, "models")
    model_files = sorted(
        [f for f in os.listdir(model_dir) if f.endswith(".keras")], reverse=True
    )
    if not model_files:
        raise FileNotFoundError("No .keras model files found in models/ directory.")
    model_path = os.path.join(model_dir, model_files[0])

    data_path = os.path.join(project_root, "data")
    print(f"🧠 Evaluating latest model: {model_files[0]}")
    evaluate_model(model_path, data_path)
    print("✅ Evaluation complete.")
