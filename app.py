import streamlit as st
from PIL import Image, ImageEnhance
import cv2
import numpy as np

st.title("AI Photo Filters & Face Enhancer ✨")
st.write("Upload your photo to apply cool AI filters!")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def smooth_skin(img):
    return cv2.bilateralFilter(img, 15, 75, 75)

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

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    np_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    st.image(image, caption="Original", width=300)

    brightness = st.slider("Brightness", 0.5, 2.0, 1.0)
    contrast = st.slider("Contrast", 0.5, 2.0, 1.0)

    if st.button("Enhance Face"):
        gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = np_img[y:y+h, x:x+w]
            np_img[y:y+h, x:x+w] = smooth_skin(roi)

        result = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
        st.image(result, caption="Smooth Skin Result", width=300)

    if st.button("Cartoon Filter"):
        cartoon_img = cartoon(np_img)
        result = cv2.cvtColor(cartoon_img, cv2.COLOR_BGR2RGB)
        st.image(result, caption="Cartoon Effect", width=300)

    # Brightness & contrast
    enhancer_b = ImageEnhance.Brightness(image)
    enhancer_c = ImageEnhance.Contrast(enhancer_b.enhance(brightness))
    filtered = enhancer_c.enhance(contrast)

    st.image(filtered, caption="Brightness/Contrast Applied", width=300)

    filtered.save("edited.png")
    with open("edited.png", "rb") as f:
        st.download_button("Download Edited Image", f, "edited.png")
