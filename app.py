import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance


st.set_page_config(page_title="AI Photo FX", layout="wide")
st.title("🎨 AI Photo FX – Smart Image Filters")


# ✅ Effects
effects = [
    "Original",
    "Pencil Sketch",
    "Ghibli Cartoon",
    "Color Pop",
    "HDR Enhance",
    "Smooth Skin"
]


# ✅ Convert uploaded image → OpenCV Format
def load_image(image):
    img = np.array(image.convert("RGB"))
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# ✅ Pencil Sketch – High-Quality
def pencil_sketch(img):
    dst_gray, _ = cv2.pencilSketch(
        img, sigma_s=70, sigma_r=0.07, shade_factor=0.05
    )
    if dst_gray.mean() < 127:
        dst_gray = cv2.bitwise_not(dst_gray)
    return dst_gray


# ✅ Anime / Ghibli-Inspired Cartoon Filter
def ghibli_cartoon(img):
    # Reduce noise but keep edges sharp
    filtered = cv2.bilateralFilter(img, 11, 100, 100)

    # Edge mask
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 9, 2
    )
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Slight saturation boost (Ghibli style)
    hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = cv2.add(hsv[..., 1], 20)
    color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cartoon = cv2.bitwise_and(color, edges)
    return cartoon


# ✅ Color Pop (Background B&W, Person Color)
def color_pop(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    mask = cv2.GaussianBlur(gray, (21, 21), 0) > 120
    result = img.copy()
    result[~mask] = gray_3ch[~mask]
    return result


# ✅ HDR Boost – Vibrant sharp look
def hdr_effect(img):
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.4)
    return hdr


# ✅ Smooth Skin
def skin_smooth(img):
    return cv2.bilateralFilter(img, 15, 80, 80)


# ✅ Auto HD Upscale
def upscale(img):
    h, w = img.shape[:2]
    scale = 1.5
    return cv2.resize(img, (int(w * scale), int(h * scale)), cv2.INTER_CUBIC)


# ✅ Convert for Streamlit display
def to_display(img):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


uploaded_image = st.file_uploader("📸 Upload image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Original", width=300)

    effect = st.selectbox("✨ Choose an Effect", effects)

    if st.button("Apply Effect"):
        img = load_image(image)

        if effect == "Original":
            result = img

        elif effect == "Pencil Sketch":
            result = pencil_sketch(img)

        elif effect == "Ghibli Cartoon":
            result = ghibli_cartoon(img)

        elif effect == "Color Pop":
            result = color_pop(img)

        elif effect == "HDR Enhance":
            result = hdr_effect(img)

        elif effect == "Smooth Skin":
            result = skin_smooth(img)

        # ✅ Final HD Enhancement
        final = upscale(result)

        display_img = to_display(final)
        st.image(display_img, caption=f"{effect} Applied ✅", width=640)

        # ✅ Download Button (HQ)
        retval, buf = cv2.imencode(".png", final)
        st.download_button(
            "⬇️ Download High-Quality PNG",
            data=buf.tobytes(),
            file_name="ai_edited.png",
            mime="image/png"
        )

st.markdown("---")
st.info("🚀 More AI Effects Coming Soon: Background remove, 4K Super resolution, AI Faces…")
