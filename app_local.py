import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="Face Detection (Local)", layout="centered")
st.title("Real-Time Face Detection & Vector Visualization (Localhost)")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def extract_feature_vector(x, y, w, h):
    return np.array([w, h, w / h if h != 0 else 0, w * h], dtype=float)

def vector_norm(v):
    return np.sqrt(np.sum(v ** 2))

class FaceVectorRT(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

            v = extract_feature_vector(x, y, w, h)
            n = vector_norm(v)

            cv2.putText(img, f"f={v.astype(int)}", (x, y-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.putText(img, f"||f||={n:.2f}", (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        return img

webrtc_streamer(
    key="face-rt",
    video_transformer_factory=FaceVectorRT,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
)
