import os
import cv2
import numpy as np
from .preprocessing import image_to_vector, normalize_vector

def load_dataset(dataset_path):
    X = []
    y = []

    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if not os.path.isdir(label_path):
            continue

        for file in os.listdir(label_path):
            img_path = os.path.join(label_path, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            vector = image_to_vector(img)
            vector = normalize_vector(vector)

            X.append(vector)
            y.append(label)

    return np.array(X), np.array(y)
