import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(data_dir="data", img_size=(96, 96), test_split=0.2):
    categories = ["non_cancerous", "cancerous"]
    data = []
    labels = []

    for label, category in enumerate(categories):
        category_path = os.path.join(data_dir, category)
        if not os.path.exists(category_path):
            print(f"⚠️  Directory not found: {category_path}")
            continue

        for file_name in os.listdir(category_path):
            if not file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue  # skip non-image files like .DS_Store

            file_path = os.path.join(category_path, file_name)
            try:
                img = cv2.imread(file_path)
                img = cv2.resize(img, img_size)
                img = img.astype("float32") / 255.0  # Normalise
                data.append(img)
                labels.append(label)
            except Exception as e:
                print(f"❌ Error loading image {file_path}: {e}")

    # Convert to NumPy arrays
    data = np.array(data)
    labels = np.array(labels)

    # Split and shuffle
    X_train, X_val, y_train, y_val = train_test_split(
        data, labels, test_size=test_split, random_state=42, stratify=labels
    )

    # ✅ Diagnostics
    print("✅ Training labels:", np.unique(y_train, return_counts=True))
    print("✅ Validation labels:", np.unique(y_val, return_counts=True))
    print("🔎 Any NaNs in X_val?", np.isnan(X_val).any())

    return X_train, X_val, y_train, y_val
