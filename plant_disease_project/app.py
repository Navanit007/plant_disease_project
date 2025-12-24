import streamlit as st
import numpy as np
import json
import tensorflow as tf
from PIL import Image
import os

# Function to load the model with error handling
def load_model():
    model_path = "models/plant_disease_recog_model_pwp.keras"
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None
    
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
model = load_model()

# Load plant disease labels with error handling
def load_labels():
    try:
        with open("plant_disease.json", 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        st.error("Label file not found: plant_disease.json")
        return []

plant_disease = load_labels()

def extract_features(image):
    """Preprocess the image for model prediction."""
    image = image.resize((160, 160))
    feature = np.array(image)
    return np.expand_dims(feature, axis=0)  # Expand dimensions to fit model input

def model_predict(image):
    """Predict the disease using the model."""
    if model is None:
        return "Model not loaded."

    img = extract_features(image)
    prediction = model.predict(img)
    predicted_index = np.argmax(prediction[0])  # Get the index of the highest probability
    prediction_label = plant_disease[predicted_index]  # Use the predicted index to get the label
    return prediction_label

# Streamlit app interface
st.title("Plant Disease Recognition")
uploaded_file = st.file_uploader("Upload an image of the plant leaf:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if st.button("Predict"):
        prediction = model_predict(image)
        st.write(f"Prediction: {prediction}")
else:
    st.write("Please upload an image to get a prediction.")
