import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch

st.set_page_config(title="AI Studio", layout="centered")
st.title("🎨 AI Photo Stylizer – Free Edition")

# Upload
file = st.file_uploader("📸 Upload your photo", type=["png","jpg","jpeg"])
style = st.selectbox("Choose AI Style", [
    "Ghibli Anime",
    "Cartoon Style",
    "Pencil Sketch",
    "HDR Effect",
    "Smooth Skin",
])

quality = st.checkbox("✨ Enhance Quality (2X Upscale)")

def enhance(image):
    return cv2.resize(image, None, fx=2, fy=2)

def pencil(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21,21), 0)
    sketch = cv2.divide(gray, 255-blur, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

def cartoon(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(blur,255,
              cv2.ADAPTIVE_THRESH_MEAN_C,
              cv2.THRESH_BINARY,9,2)
    color = cv2.bilateralFilter(img,9,300,300)
    return cv2.bitwise_and(color, color, mask=edges)

def hdr(img):
    return cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)

def smooth(img):
    return cv2.bilateralFilter(img, 17, 75, 75)

process = None

if file:
    image = np.array(Image.open(file).convert("RGB"))
    show = image.copy()

    if st.button("✨ Transform Now"):
        with st.spinner("AI Processing... ⏳"):
            if style == "Pencil Sketch":
                process = pencil(show)

            elif style == "Cartoon Style":
                process = cartoon(show)

            elif style == "HDR Effect":
                process = hdr(show)

            elif style == "Smooth Skin":
                process = smooth(show)

            elif style == "Ghibli Anime":
                process = cartoon(show)

        if quality:
            process = enhance(process)

        st.image(process, caption="AI Output", channels="RGB")

        # Prepare download
        result = Image.fromarray(process)
        st.download_button(
            "⬇ Download Result",
            data=cv2.imencode(".png", process)[1].tobytes(),
            file_name="ai_result.png",
            mime="image/png"
        )
