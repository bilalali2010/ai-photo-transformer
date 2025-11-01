import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI Studio", layout="wide")
st.title("🎨 AI Photo Stylizer – Free Edition")

# Upload
file = st.file_uploader("📸 Upload your photo", type=["png", "jpg", "jpeg"])
style = st.selectbox("Choose AI Style", [
    "Ghibli Anime",
    "Cartoon Style",
    "Pencil Sketch",
    "HDR Effect",
    "Smooth Skin",
])

quality = st.checkbox("✨ Enhance Quality (2X Upscale)")

def enhance(image):
    return cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

def pencil(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21,21), 0)
    sketch = cv2.divide(gray, 255-blur, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

def cartoon(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(blur,255,
              cv2.ADAPTIVE_THRESH_MEAN_C,
              cv2.THRESH_BINARY,9,3)
    color = cv2.bilateralFilter(img,9,300,300)
    return cv2.bitwise_and(color, color, mask=edges)

def hdr(img):
    return cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)

def smooth(img):
    return cv2.bilateralFilter(img, 17, 75, 75)

process = None

if file:
    image = np.array(Image.open(file).convert("RGB"))
    show = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

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
                process = cartoon(show)  # placeholder - upgrade soon ✅

        if quality:
            process = enhance(process)

        result_rgb = cv2.cvtColor(process, cv2.COLOR_BGR2RGB)
        st.image(result_rgb, caption="✅ Transformed Output", channels="RGB")

        st.download_button(
            "⬇ Download Result",
            data=cv2.imencode(".png", result_rgb)[1].tobytes(),
            file_name="ai_result.png",
            mime="image/png"
        )
