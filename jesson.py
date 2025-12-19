import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# -------- PARAMETERS --------
IMAGE_PATH = "image-compressable.jpg"   # change to your image filename
k = 100                     # compression level
# ----------------------------


def compress_image_svd(image_path, k):
    image = Image.open(image_path).convert('L')  # grayscale
    A = np.array(image, dtype=float)

    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    # Keep only k singular values
    A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

    return A, A_k


def main():
    original, compressed = compress_image_svd(IMAGE_PATH, k)

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"Compressed Image (k = {k})")
    plt.imshow(compressed, cmap="gray")
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()