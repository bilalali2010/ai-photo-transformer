import streamlit as st
from PIL import Image, ImageEnhance
import cv2
import numpy as np

st.set_page_config(page_title="AI Photo Studio", layout="centered")
st.title("🎨 AI Photo Studio – Face Filters & Enhancer")

uploaded_file = st.file_uploader("Upload a clear face image", type=["jpg", "jpeg", "png"])

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def smooth_skin(img):
    return cv2.bilateralFilter(img, 15, 75, 75)

def sharpen(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def cartoon(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(
        cv2.medianBlur(gray, 5),
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        9,
        9
    )
    color = cv2.bilateralFilter(img, 10, 250, 250)
    return cv2.bitwise_and(color, color, mask=edges)

def background_blur(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for (x, y, w, h) in faces:
        mask[y:y+h, x:x+w] = 255

    blurred_img = cv2.GaussianBlur(img, (51, 51), 0)
    return np.where(mask[:, :, None] == 255, img, blurred_img)

def beautify_face(img):
    smoothed = smooth_skin(img)
    return sharpen(smoothed)

def pencil_sketch(img):
    gray, sketch = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.5, shade_factor=0.02)
    return sketch

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    np_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    st.image(image, caption="Original", width=300)

    filter_choice = st.selectbox(
        "Choose an effect",
        [
            "Smooth Skin",
            "Sharpen",
            "Cartoon",
            "Background Blur",
            "Black & White",
            "Sepia",
            "Pencil Sketch"
        ]
    )

    if st.button("Apply Effect"):
        if filter_choice == "Smooth Skin":
            result = beautify_face(np_img)
        elif filter_choice == "Sharpen":
            result = sharpen(np_img)
        elif filter_choice == "Cartoon":
            result = cartoon(np_img)
        elif filter_choice == "Background Blur":
            result = background_blur(np_img)
        elif filter_choice == "Black & White":
            result = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
        elif filter_choice == "Sepia":
            kernel = np.array([[0.272, 0.534, 0.131],
                               [0.349, 0.686, 0.168],
                               [0.393, 0.769, 0.189]])
            result = cv2.transform(np_img, kernel)
            result = np.clip(result, 0, 255)
        elif filter_choice == "Pencil Sketch":
            result = pencil_sketch(np_img)

        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB) if len(result.shape) == 3 else result
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

        st.image(result, caption=f"{filter_choice} Applied ✅", width=300)

        result_img = Image.fromarray(result)
        result_img.save("edited.png")
        with open("edited.png", "rb") as f:
            st.download_button("Download Image ✅", f, "ai_edited.png")
