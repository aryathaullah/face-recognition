import cv2
import numpy as np

def extract_feature_vector(w, h):
    return np.array([w, h, w / h if h != 0 else 0, w * h], dtype=float)

def vector_norm(v):
    return np.sqrt(np.sum(v ** 2))

def run_video_mode(st):
    
    video_file = st.file_uploader(
        "UPLOAD VIDEO (MP4, AVI, MOV)",
        type=["mp4", "avi", "mov"]
    )

    if video_file:
        with open("temp_video.mp4", "wb") as f:
            f.write(video_file.read())

        cap = cv2.VideoCapture("temp_video.mp4")
        frame_box = st.empty()

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

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
