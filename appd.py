import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Page configure
st.set_page_config(page_title="Anime Face Generator", page_icon="🎨")

# Function to load model
@st.cache_resource
def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error in loading the model : {e}")
        return None

# Siderbar Control
st.sidebar.header("Model settings")
model_choice = st.sidebar.selectbox(
    "Select the model format",
    [".keras", ".h5"],
    index=0
)

latent_dim = 100
num_images = st.sidebar.slider("Choose number of images", 1, 25, 9)
seed = st.sidebar.number_input("Choose random seed value", value=None,step = 1)

# Main app
st.title("Anime Face Generator DCGAN")
st.write("Make your own AI generated anime")

if st.sidebar.button("Generate Images"):
    # Load Model path
    model_path = f"models/generator{model_choice}"
    
    # Load Model
    with st.spinner("Model is loading"):
        generator = load_model(model_path)
    
    if generator is not None:
        # Generating random noise
        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)
        
        noise = tf.random.normal((num_images, latent_dim))
        
        # Generate Images
        with st.spinner("Images are generating "):
            generated_images = generator(noise, training=False)
            generated_images = (generated_images * 127.5) + 127.5  # [-1,1] से [0,255]
            generated_images = generated_images.numpy().astype(np.uint8)
        
        # Show images in grid format
        st.subheader("Here are your Generated Images")
        cols = 3
        rows = (num_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(10, 2*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_images):
            row = i // cols
            col = i % cols
            img = Image.fromarray(generated_images[i])
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
        
        # Hide vacant subplots
        for i in range(num_images, rows*cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Download button
        if num_images == 1:
            img = Image.fromarray(generated_images[0])
            st.download_button(
                label="Download the image",
                data=img.tobytes(),
                file_name="generated_anime.jpg",
                mime="image/jpg"
            )

#  Information in side bar
st.sidebar.header("Information")
st.sidebar.info("""
In this app DCGAN (Deep Convolutional Generative Adversarial Network) uses for 
64x64 pixel image genration
""")