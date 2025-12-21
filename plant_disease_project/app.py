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
st.set_page_config(page_title="Plant Disease Detection", layout="centered")

# -----------------------
# Dropbox model link
# -----------------------
MODEL_PATH = "plant_disease_recog_model_pwp.keras"
MODEL_URL = "https://dl.dropboxusercontent.com/s/agj0djj2oqa7zl9okw51n/plant_disease_recog_model_pwp.keras"

# -----------------------
# Load model (cached)
# -----------------------
@st.cache_resource
def load_model():
    # Download model if not exists
    if not os.path.exists(MODEL_PATH):
        st.write("⬇️ Downloading model from Dropbox...")
        try:
            r = requests.get(MODEL_URL, stream=True)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            st.write("✅ Model downloaded.")
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            st.stop()
    st.write("✅ Loading model...")
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# -----------------------
# Load disease info
# -----------------------
with open("plant_disease.json", "r") as file:
    plant_disease = json.load(file)

# -----------------------
# Labels (optional if already in JSON)
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
    image = image.convert("RGB")  # ensure 3 channels
    image = image.resize((160, 160))
    image = np.array(image) / 255.0  # normalize pixels
    image = np.expand_dims(image, axis=0)
    return image

# -----------------------
# Prediction function
# -----------------------
def predict_disease(image):
    img = preprocess_image(image)
    prediction = model.predict(img)
    index = np.argmax(prediction)
    # Return description from JSON if exists, else label
    return plant_disease.get(str(index), labels[index])

# -----------------------
# Streamlit UI
# -----------------------
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
