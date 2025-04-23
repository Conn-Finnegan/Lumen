import os
import sys

# Add the 'src' folder to Python's module search path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from load_data import load_data
from model import build_model


def main():
    # Set data path relative to the project root
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(project_root, "data")

    # Load and split the dataset
    X_train, X_val, y_train, y_val = load_data(data_dir=data_path)

    print("Training set shape:", X_train.shape, y_train.shape)
    print("Validation set shape:", X_val.shape, y_val.shape)

    # Build the CNN model
    model = build_model(input_shape=(50, 50, 3))
    model.summary()


if __name__ == "__main__":
    main()
