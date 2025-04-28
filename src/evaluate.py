import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model_path, val_dir):
    # Load model
    model = load_model(model_path)

    # Prepare validation generator
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(96, 96),  # Make sure it matches training
        batch_size=32,
        class_mode="binary",
        shuffle=False,
    )

    # Predict
    y_pred_probs = model.predict(val_generator, verbose=1)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    # True labels
    y_true = val_generator.classes

    # Print classification report
    print("\n📊 Classification Report:")
    report = classification_report(
        y_true, y_pred, target_names=["Non-cancerous", "Cancerous"]
    )
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-cancerous", "Cancerous"],
        yticklabels=["Non-cancerous", "Cancerous"],
    )
    plt.title("Confusion Matrix on Validation Set")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    # Save confusion matrix
    os.makedirs("outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    plot_path = f"outputs/confusion_matrix_{timestamp}.png"
    plt.savefig(plot_path)
    plt.show()

    print(f"📷 Confusion matrix saved to: {plot_path}")


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(__file__))

    # Find latest model automatically
    model_dir = os.path.join(project_root, "models")
    model_files = sorted(
        [f for f in os.listdir(model_dir) if f.endswith(".keras")], reverse=True
    )
    if not model_files:
        raise FileNotFoundError("No .keras model files found in models/ directory.")
    model_path = os.path.join(model_dir, model_files[0])

    val_path = os.path.join(project_root, "data", "val")

    print(f"🧠 Evaluating latest model: {model_files[0]}")
    evaluate_model(model_path, val_path)
    print("✅ Evaluation complete.")
