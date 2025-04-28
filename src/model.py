from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    Input,
)
from tensorflow.keras.regularizers import l2


def build_model(input_shape=(50, 50, 3)):
    model = Sequential()

    # Input layer
    model.add(Input(shape=input_shape))

    # Block 1
    model.add(
        Conv2D(
            32, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001)
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 2
    model.add(
        Conv2D(
            64, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001)
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 3
    model.add(
        Conv2D(
            128, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001)
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 4 (new deeper block)
    model.add(
        Conv2D(
            128, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001)
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 5 (extra depth)
    model.add(
        Conv2D(
            256, (3, 3), activation="relu", padding="same", kernel_regularizer=l2(0.001)
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten and Dense Layers
    model.add(Flatten())
    model.add(Dense(256, activation="relu", kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model
