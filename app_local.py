import streamlit as st
from streamlit_webrtc import webrtc_streamer
from src.camera.camera_feature import FaceVectorRT

st.set_page_config(page_title="Face Detection (Local)", layout="centered")
st.title("CAMERA-REALTIME FACE DETECTION (LOCALHOST)")

webrtc_streamer(
    key="face-camera",
    video_transformer_factory=FaceVectorRT,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True
)
