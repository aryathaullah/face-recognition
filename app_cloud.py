import streamlit as st
from src.images_based.image_feature import run_image_mode
from src.videos_based.video_feature import run_video_mode

st.set_page_config(page_title="Face Detection (Cloud)", layout="centered")

with st.sidebar:
    menu = st.selectbox(
        "SELECT MODE",
        options=["IMAGES-BASED", "VIDEO-BASED", "CAMERA-BASED"]
    )

if menu == "IMAGES-BASED":
    st.title("IMAGE - FACE DETECTION")
    run_image_mode(st)
elif menu == "VIDEO-BASED":
    st.title("VIDEO - FACE DETECTION")
    run_video_mode(st)
elif menu == "CAMERA-BASED":
    st.title("CAMERA - FACE DETECTION")
    st.info("camera-based face detection is only available in the LOCALHOST mode. "
            "Please switch to the LOCALHOST app to use this feature.")
    st.info("https://github.com/aryathaullah/face-recognition.git for more info.")