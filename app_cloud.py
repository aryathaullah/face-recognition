import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(page_title="Face Detection (Cloud)", layout="centered")
st.title("Face Detection & Vector Visualization (Cloud Version)")

st.sidebar.title("Menu")
menu = st.sidebar.radio(
    "Choose Mode",
    ["Image", "Video", "Real-Time Camera (localhost only)"]
)

st.write("✅ Cloud-safe mode (no camera access)")

def load_face_detector():
    import cv2
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

def extract_feature_vector(w, h):
    return np.array([w, h, w / h if h != 0 else 0, w * h], dtype=float)

def vector_norm(v):
    return np.sqrt(np.sum(v ** 2))

# =========================
# IMAGE MODE
# =========================
if menu == "Image":
    st.header("🖼️ Image Face Detection")

    img_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

    if img_file:
        import cv2

        image = Image.open(img_file).convert("RGB")
        img = np.array(image)

        face_cascade = load_face_detector()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

            v = extract_feature_vector(w, h)
            n = vector_norm(v)

            cv2.putText(img, f"f={v.astype(int)}", (x, y-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.putText(img, f"||f||={n:.2f}", (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        st.image(img, use_column_width=True)

# =========================
# VIDEO MODE
# =========================
elif menu == "Video":
    st.header("🎞️ Video Face Detection")

    video_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])

    if video_file:
        import cv2

        with open("temp_video.mp4", "wb") as f:
            f.write(video_file.read())

        cap = cv2.VideoCapture("temp_video.mp4")
        frame_box = st.empty()

        face_cascade = load_face_detector()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

                v = extract_feature_vector(w, h)
                n = vector_norm(v)

                cv2.putText(frame, f"||f||={n:.2f}", (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            frame_box.image(frame, channels="BGR", use_column_width=True)

        cap.release()

elif menu == "Real-Time Camera (localhost only)":
    st.header("📷 Real-Time Camera")

    st.warning(
        "This feature ONLY works on localhost.\n"
        "Streamlit Community Cloud does NOT support camera access."
    )

    st.code(
        "Run this app locally using:\n\n"
        "streamlit run app.py",
        language="bash"
    )
