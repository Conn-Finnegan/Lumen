import os
from load_data import load_data
from model import build_model
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model


def train(
    data_dir="data", save_path="models/cancer_classifier.h5", epochs=10, batch_size=64
):
    # Load the data
    X_train, X_val, y_train, y_val = load_data(data_dir=data_dir)

    # Build the model
    model = build_model(input_shape=(50, 50, 3))

    # Create a directory for saved models if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the best model based on validation accuracy
    checkpoint = ModelCheckpoint(
        filepath=save_path, monitor="val_accuracy", save_best_only=True, verbose=1
    )

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint],
    )

    # Plot accuracy and loss
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/training_plot.png")
    plt.show()

    print(f"✅ Training complete. Best model saved to {save_path}")


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(project_root, "data")
    model_path = os.path.join(project_root, "models/cancer_classifier.h5")
    train(data_dir=data_path, save_path=model_path, epochs=10, batch_size=64)
