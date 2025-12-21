import streamlit as st
import numpy as np
import tensorflow as tf
import json
from PIL import Image
import os
import requests

# -----------------------
# Page config
# -----------------------
st.set_page_config(
    page_title="Plant Disease Detection",
    layout="centered"
)

# -----------------------
# Hugging Face model link (IMPORTANT)
# -----------------------
MODEL_PATH = "plant_disease_recog_model_pwp.keras"
MODEL_URL = (
    "https://huggingface.co/Navanit007/plant-disease-model/"
    "resolve/main/plant_disease_recog_model_pwp.keras"
)

# -----------------------
# Load model (cached)
# -----------------------
@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("⬇️ Downloading model from Hugging Face...")
        r = requests.get(MODEL_URL, stream=True, timeout=60)
        r.raise_for_status()

        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)

        st.success("✅ Model downloaded successfully")

    st.info("📦 Loading model (first time may take ~30s)...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    st.success("✅ Model loaded")
    return model

model = load_model()

# -----------------------
# Load disease info
# -----------------------
with open("plant_disease.json", "r") as file:
    plant_disease = json.load(file)

# -----------------------
# Labels
# -----------------------
labels = [
    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust',
    'Apple___healthy','Background_without_leaves','Blueberry___healthy',
    'Cherry___Powdery_mildew','Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot','Corn___Common_rust',
    'Corn___Northern_Leaf_Blight','Corn___healthy',
    'Grape___Black_rot','Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot','Peach___healthy',
    'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy',
    'Potato___Early_blight','Potato___Late_blight','Potato___healthy',
    'Raspberry___healthy','Soybean___healthy',
    'Squash___Powdery_mildew','Strawberry___Leaf_scorch',
    'Strawberry___healthy','Tomato___Bacterial_spot',
    'Tomato___Early_blight','Tomato___Late_blight',
    'Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus','Tomato___healthy'
]

# -----------------------
# Image preprocessing
# -----------------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((160, 160))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -----------------------
# Prediction
# -----------------------
def predict_disease(image):
    img = preprocess_image(image)
    prediction = model.predict(img, verbose=0)
    index = int(np.argmax(prediction))
    return plant_disease.get(str(index), labels[index])

# -----------------------
# UI
# -----------------------
st.title("🌱 Plant Disease Detection")
st.write("Upload a leaf image to detect plant disease")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict Disease"):
        with st.spinner("🔍 Analyzing image..."):
            result = predict_disease(image)
        st.success("✅ Prediction Complete")
        st.subheader("🦠 Disease Information")
        st.write(result)
