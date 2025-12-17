import cv2
import numpy as np

def image_to_vector(image, size=(100, 100)):
    """
    Convert image to vector (Linear Algebra representation)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, size)
    vector = resized.flatten().astype(np.float64)
    return vector

def normalize_vector(v):
    """
    Vector normalization using Euclidean norm
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm
