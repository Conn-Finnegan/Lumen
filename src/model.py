from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def build_model(input_shape=(50, 50, 3)):
    """
    Builds a simple CNN for binary classification of histopathology images.

    Parameters:
        input_shape (tuple): Shape of input images (default is (50, 50, 3))

    Returns:
        model (keras.Model): Compiled CNN model
    """
    model = Sequential()

    # First Convolutional Block
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Second Convolutional Block
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Third Convolutional Block
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten and Dense Layers
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))  # Dropout to reduce overfitting
    model.add(Dense(1, activation="sigmoid"))  # Binary output

    # Compile the model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model
