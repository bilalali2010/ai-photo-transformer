import streamlit as st
from PIL import Image, ImageEnhance
import cv2
import numpy as np
from gfpgan import GFPGANer

st.title("AI Photo Enhancer & Face Beautifier ✨")

@st.cache_resource
def load_model():
    return GFPGANer(
        model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.8/GFPGANv1.4.pth",
        upscale=2
    )

gfpgan = load_model()

uploaded_file = st.file_uploader("Upload a face photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.image(image, caption="Original", width=300)

    if st.button("Enhance Face"):
        with st.spinner("Improving face quality..."):
            _, _, enhanced_face = gfpgan.enhance(img_np, has_aligned=False)
            enhanced_img = Image.fromarray(enhanced_face)
            st.image(enhanced_img, caption="Enhanced Image", width=300)

            enhanced_img.save("enhanced.png")
            with open("enhanced.png", "rb") as f:
                st.download_button("Download Result", f, "enhanced.png")

    brightness = st.slider("Brightness", 0.5, 2.0, 1.0)
    contrast = st.slider("Contrast", 0.5, 2.0, 1.0)

    if st.button("Apply Filters"):
        enhancer_b = ImageEnhance.Brightness(image)
        enhancer_c = ImageEnhance.Contrast(enhancer_b.enhance(brightness))
        filtered = enhancer_c.enhance(contrast)

        st.image(filtered, caption="Filtered Image", width=300)

        filtered.save("filtered.png")
        with open("filtered.png", "rb") as f:
            st.download_button("Download Filtered", f, "filtered.png")
