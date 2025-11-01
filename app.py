import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ✅ UI Configuration
st.set_page_config(page_title="AI Photo Studio", layout="wide")

# ✅ Custom Modern White UI Styling
white_ui = """
<style>
.stApp {
    background: #F6F9FC !important;
}
h1, h2, h3 {
    color: #222 !important;
    font-weight: 700 !important;
}
label, p, .stMarkdown {
    color: #333 !important;
}
.stButton button {
    background: linear-gradient(90deg, #4A90E2, #1EA7FD);
    color: white !important;
    border: none;
    padding: 12px 22px;
    border-radius: 10px;
    font-size: 17px;
    font-weight: 600;
    transition: 0.25s;
}
.stButton button:hover {
    box-shadow: 0px 4px 20px rgba(30,167,253,0.4);
}
.sidebar .sidebar-content {
    background-color: #ffffff !important;
    border-right: 1px solid #E4E8EC;
}
img {
    border-radius: 12px;
}
.selectbox label, .file-uploader label {
    font-size: 16px;
    font-weight: 600;
}
</style>
"""
st.markdown(white_ui, unsafe_allow_html=True)

st.title("✨ AI Photo Studio")
st.caption("Transform your photos into creative art using AI — free & fast!")

file = st.file_uploader("📸 Upload your image", type=["png", "jpg", "jpeg"])

style = st.selectbox("🎨 Choose Art Style", [
    "Ghibli Anime",
    "Cartoon Style",
    "Pencil Sketch",
    "HDR Effect",
    "Smooth Skin",
])

quality = st.checkbox("🔹 Enhance Quality (HD Upscale)")

# ✅ Effect Functions
def enhance(img):
    return cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

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
    color = cv2.bilateralFilter(img,9,290,290)
    return cv2.bitwise_and(color, color, mask=edges)

def hdr(img):
    return cv2.detailEnhance(img, sigma_s=10, sigma_r=0.2)

def smooth(img):
    return cv2.bilateralFilter(img, 25, 40, 40)

# ✅ Main UI Logic
if file:
    img = np.array(Image.open(file).convert("RGB"))
    show = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📌 Original Image")
        st.image(img, use_column_width=True)

    if st.button("✨ Transform Now"):
        with st.spinner("Processing with AI... 🚀"):
            if style == "Pencil Sketch":
                out = pencil(show)

            elif style == "Cartoon Style":
                out = cartoon(show)

            elif style == "HDR Effect":
                out = hdr(show)

            elif style == "Smooth Skin":
                out = smooth(show)

            elif style == "Ghibli Anime":
                out = cartoon(show)  # Placeholder (anime model upgrade coming)

        if quality:
            out = enhance(out)

        output = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

        with col2:
            st.subheader("✅ AI Result")
            st.image(output, use_column_width=True)

        st.download_button(
            "⬇ Download Artwork",
            data=cv2.imencode(".png", output)[1].tobytes(),
            file_name="AI_Art.png",
            mime="image/png"
        )
