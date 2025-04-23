import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(data_dir="data", img_size=50, test_split=0.2):
    X = []
    y = []

    label_map = {"non_cancerous": 0, "cancerous": 1}

    for folder, label in label_map.items():
        folder_path = os.path.join(data_dir, folder)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (img_size, img_size))
                X.append(img)
                y.append(label)

    X = np.array(X, dtype="float32") / 255.0
    y = np.array(y)

    return train_test_split(X, y, test_size=test_split, stratify=y, random_state=42)
