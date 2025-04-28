import os
import matplotlib.pyplot as plt
from datetime import datetime
from load_data import load_data
from model import build_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def train(data_dir="data", epochs=50, batch_size=64):
    # Load data
    X_train, X_val, y_train, y_val = load_data(data_dir=data_dir)

    # Light Data augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.9, 1.1],
    )

    val_datagen = ImageDataGenerator()

    # Create generators
    train_generator = train_datagen.flow(
        X_train, y_train, batch_size=batch_size, shuffle=True
    )
    val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size, shuffle=False)

    # Build model
    model = build_model(input_shape=(50, 50, 3))

    # Timestamped model saving
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_filename = f"cancer_classifier_{timestamp}.keras"
    save_path = os.path.join("models", save_filename)

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Callbacks
    checkpoint = ModelCheckpoint(
        filepath=save_path, monitor="val_accuracy", save_best_only=True, verbose=1
    )

    early_stop = EarlyStopping(
        monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1
    )

    # Class weights (still slightly favour cancerous)
    class_weight = {0: 1.0, 1: 1.3}

    # Train model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        steps_per_epoch=len(train_generator),
        validation_steps=len(val_generator),
        epochs=epochs,
        callbacks=[checkpoint, early_stop],
        class_weight=class_weight,
        verbose=1,
    )

    # Plot training history
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.title("Model Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Model Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Save plot
    plot_path = f"outputs/training_plot_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()

    print(f"✅ Training complete.")
    print(f"📦 Model saved to: {save_path}")
    print(f"📈 Plot saved to: {plot_path}")


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(project_root, "data")
    train(data_dir=data_path, epochs=50, batch_size=64)
