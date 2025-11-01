import streamlit as st
import cv2
import numpy as np
from PIL import Image

# -----------------------------------------------------------
# ✅ Page Setup & Custom Styling
# -----------------------------------------------------------
st.set_page_config(page_title="AI Photo Studio", layout="wide")

page_bg = """
<style>
body {
    background-color: #0D0D0D;
}
.css-1v0mbdj, .stApp {
    background: #0D0D0D !important;
}
h1, h2, h3, label, p, .stMarkdown {
    color: #E0E0E0 !important;
}
.stButton button {
    background: linear-gradient(90deg, #007BFF, #00EEFF);
    color: white !important;
    border: none;
    padding: 10px 20px;
    border-radius: 8px;
    transition: .3s;
    font-size: 17px;
    font-weight: 600;
}
.stButton button:hover {
    box-shadow: 0px 0px 15px #00E7FF;
    cursor: pointer;
}
.sidebar .sidebar-content {
    background: #111111;
}
.uploadedFile {
    border-radius: 10px !important;
}
.result-card {
    padding: 10px;
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.10);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

st.title("🎨 AI Photo Studio")
st.write("Turn Your Photo into a Stunning Artwork — Free & Fast 🚀")

# Upload
file = st.file_uploader("📸 Upload Image", type=["png", "jpg", "jpeg"])

style = st.selectbox("✨ Select AI Style", [
    "Ghibli Anime",
    "Cartoon Style",
    "Pencil Sketch",
    "HDR Effect",
    "Smooth Skin",
])

quality = st.checkbox("🔹 Enhance Quality (HD Upscale)")

# -----------------------------------------------------------
# ✅ Style Functions
# -----------------------------------------------------------
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
    return cv2.detailEnhance(img, sigma_s=10, sigma_r=0.25)

def smooth(img):
    return cv2.bilateralFilter(img, 25, 50, 50)

# -----------------------------------------------------------
# ✅ Processing + UI Result View
# -----------------------------------------------------------
if file:
    image = np.array(Image.open(file).convert("RGB"))
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📌 Original")
        st.image(image, use_column_width=True)

    if st.button("⚡ Transform Now"):
        with st.spinner("🎨 Creating AI Magic..."):
            if style == "Pencil Sketch":
                output = pencil(img_bgr)
            elif style == "Cartoon Style":
                output = cartoon(img_bgr)
            elif style == "HDR Effect":
                output = hdr(img_bgr)
            elif style == "Smooth Skin":
                output = smooth(img_bgr)
            elif style == "Ghibli Anime":
                output = cartoon(img_bgr)  # Will upgrade to true anime soon ✅

        if quality:
            output = enhance(output)

        result_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

        with col2:
            st.subheader("✅ AI Result")
            st.image(result_rgb, use_column_width=True)

        st.download_button(
            "⬇ Download Artwork",
            data=cv2.imencode(".png", result_rgb)[1].tobytes(),
            file_name="AI_Stylized.png",
            mime="image/png"
        )
