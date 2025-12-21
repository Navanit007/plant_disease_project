import gdown
import streamlit as st
import numpy as np
import tensorflow as tf
import json
import os
from PIL import Image


# Page config
st.set_page_config(page_title="Plant Disease Detection", layout="centered")

# --- Model download settings ---
MODEL_PATH = "plant_disease_recog_model_pwp.keras"
DRIVE_ID = "1qg_5Fz1w6xfzGljce3fDv8s_DgcpEVC1"
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_ID}"

# Load model (cached for performance)
@st.cache_resource
def load_model():
    # Download model only if not present
    if not os.path.exists(MODEL_PATH):
        st.write("⬇️ Downloading model from Google Drive...")
        gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)

    st.write("✅ Loading model...")
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Load disease info
with open("plant_disease.json", "r") as file:
    plant_disease = json.load(file)

# Image preprocessing
def preprocess_image(image):
    image = image.resize((160, 160))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image

# Prediction function
def predict_disease(image):
    img = preprocess_image(image)
    prediction = model.predict(img)
    index = np.argmax(prediction)
    return plant_disease[index]

# UI
st.title("🌱 Plant Disease Detection")
st.write("Upload a leaf image to detect plant disease")

uploaded_file = st.file_uploader(
    "Choose an image", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict Disease"):
        with st.spinner("Analyzing image..."):
            result = predict_disease(image)

        st.success("Prediction Complete")
        st.subheader("🦠 Disease Information")
        st.write(result)

