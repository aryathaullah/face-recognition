import numpy as np

def euclidean_distance(x, y):
    return np.linalg.norm(x - y)

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
