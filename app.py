import streamlit as st
import cv2
import numpy as np
from PIL import Image

from src.loader import load_dataset
from src.preprocessing import image_to_vector, normalize_vector
from src.recognizer import recognize

st.set_page_config(page_title="Face Recognition - Linear Algebra", layout="centered")

st.title("Implementation of Linear Algebra Concepts in Face Recognition")
st.write("Face recognition using **vector space representation and distance metrics**")

# Load dataset
with st.spinner("Loading dataset..."):
    X_train, y_train = load_dataset("data/train")

st.success(f"Dataset loaded: {len(X_train)} images")

# Upload image
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "png", "jpeg"])

method = st.selectbox(
    "Select Distance Metric",
    ["euclidean", "cosine"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    img = np.array(image)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    vector = image_to_vector(img)
    vector = normalize_vector(vector)

    if st.button("Recognize Face"):
        label, score = recognize(vector, X_train, y_train, method)

        st.subheader("Recognition Result")
        st.write(f"**Predicted Identity:** {label}")
        st.write(f"**Similarity Score:** {score:.4f}")
