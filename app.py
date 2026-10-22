import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

# Configure the application
st.set_page_config(
    page_title="Face Detection Application",
    page_icon="👤",
    layout="centered",
)

# Application heading
st.title("Face Detection Application")
st.write("Upload an image or take a photo to detect faces using OpenCV.")


@st.cache_resource
def load_detector():
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(path)
    if detector.empty():
        raise RuntimeError("Could not load Haar cascade file.")
    return detector


def detect_faces(image_rgb, scale_factor, min_neighbors, min_size):
    detector = load_detector()
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_size, min_size),
    )
    output = image_rgb.copy()
    thickness = max(2, image_rgb.shape[1] // 300)
    for (x, y, w, h) in faces:
        cv2.rectangle(output, (x, y), (x + w, y + h), (56, 189, 248), thickness)
    return output, len(faces)


# Sidebar settings
st.sidebar.header("Detection settings")
scale_factor = st.sidebar.slider("Scale factor", 1.05, 1.5, 1.1, 0.05)
min_neighbors = st.sidebar.slider("Min neighbors", 1, 10, 5)
min_size = st.sidebar.slider("Min face size (px)", 20, 200, 30, 5)

# Input source
source = st.radio("Choose input", ["Upload image", "Use camera"], horizontal=True)

if source == "Upload image":
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
else:
    file = st.camera_input("Take a photo")

if file is not None:
    image = Image.open(file).convert("RGB")
    image_np = np.array(image)

    result, count = detect_faces(image_np, scale_factor, min_neighbors, min_size)

    if count == 0:
        st.warning("No faces detected. Try lowering Min neighbors or Min face size.")
    else:
        st.success(f"{count} face{'s' if count != 1 else ''} detected")

    col1, col2 = st.columns(2)
    col1.image(image, caption="Original", use_container_width=True)
    col2.image(result, caption="Detected faces", use_container_width=True)

    buffer = BytesIO()
    Image.fromarray(result).save(buffer, format="PNG")
    st.download_button(
        "Download result",
        data=buffer.getvalue(),
        file_name="detected_faces.png",
        mime="image/png",
    )