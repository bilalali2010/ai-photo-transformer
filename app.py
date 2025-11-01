import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline

st.title("AI Photo Transformer 🔮")
st.write("Upload your image and choose a style!")

# Load model
@st.cache_resource
def load_model():
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32
    )
    pipe = pipe.to("cpu")  # free hosting uses CPU
    return pipe

pipe = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

styles = {
    "Cartoon": "make this face look like a cartoon character",
    "Anime": "turn this person into anime style with big eyes and smooth skin",
    "Professional Headshot": "make this look like a high-quality professional portrait photo",
    "Old Age": "make this person look 60 years old",
    "Baby Face": "make this person look young like a child"
}

style_choice = st.selectbox("Choose a transformation", list(styles.keys()))

strength = st.slider("Transformation Strength", 0.1, 1.0, 0.7)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", width=300)

    if st.button("Transform"):
        with st.spinner("Generating... This may take 30–50 seconds."):
            prompt = styles[style_choice]
            result = pipe(prompt=prompt, image=image, strength=strength, guidance_scale=7.5)
            output = result.images[0]
            st.image(output, caption="Transformed", width=300)

            output.save("output.png")
            with open("output.png", "rb") as f:
                st.download_button("Download Image", f, "transformed.png")
