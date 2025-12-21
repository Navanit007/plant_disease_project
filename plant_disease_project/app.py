import os
import time
import re
import requests
import streamlit as st
import numpy as np
import tensorflow as tf
import json
from PIL import Image

# Make Streamlit bind to Render's PORT if present (optional)
os.environ.setdefault("STREAMLIT_SERVER_PORT", os.getenv("PORT", "8501"))
os.environ.setdefault("STREAMLIT_SERVER_ADDRESS", "0.0.0.0")

st.set_page_config(page_title="Plant Disease Detection", layout="centered")

MODEL_PATH = "plant_disease_recog_model_pwp.keras"
# Make sure this FILE_ID is correct and the file is shared "anyone with the link"
FILE_ID = "1rcnIg1vCj6BwTu0YflFxdkMwjdjAiB57"

def download_file_from_google_drive(file_id, destination, max_retries=3):
    """
    Robust download for Google Drive large file links:
    - Uses cookies 'download_warning' confirm token if present
    - Retries up to max_retries
    """
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    for attempt in range(1, max_retries + 1):
        print(f"[download] Attempt {attempt} for file_id={file_id}")
        try:
            response = session.get(URL, params={'id': file_id}, stream=True, timeout=60)
        except Exception as e:
            print(f"[download] Request error: {e}")
            if attempt < max_retries:
                time.sleep(2 * attempt)
                continue
            raise

        token = None
        # Check cookies for confirmation token
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                token = value
                break

        # If not in cookies, sometimes the confirm code is in the HTML form (extract)
        if not token:
            text = response.text
            # pattern like confirm=ABC123
            m = re.search(r"confirm=([0-9A-Za-z-_]+)&", text)
            if m:
                token = m.group(1)

        if token:
            print(f"[download] Confirm token found: {token[:8]}...")
            params = {'id': file_id, 'confirm': token}
            response = session.get(URL, params=params, stream=True, timeout=60)

        if response.status_code != 200:
            print(f"[download] Bad status code: {response.status_code}")
            if attempt < max_retries:
                time.sleep(2 * attempt)
                continue
            response.raise_for_status()

        # Write file
        try:
            with open(destination, "wb") as f:
                for chunk in response.iter_content(32768):
                    if chunk:
                        f.write(chunk)
        except Exception as e:
            print(f"[download] Write error: {e}")
            if attempt < max_retries:
                time.sleep(2 * attempt)
                continue
            raise

        # Quick sanity check: file size
        size = os.path.getsize(destination)
        if size == 0:
            print("[download] Warning: file size 0 after download")
            if attempt < max_retries:
                time.sleep(2 * attempt)
                continue
            raise RuntimeError("Downloaded file has size 0")
        print(f"[download] Completed. File saved to {destination} ({size} bytes)")
        return

    raise RuntimeError("Failed to download file after retries")

@st.cache_resource
def load_model():
    # Logging to stdout appears in Render logs
    print(f"[load_model] Checking model at {MODEL_PATH} (cwd={os.getcwd()})")
    if not os.path.exists(MODEL_PATH):
        st.info("⬇️ Model not found — attempting to download...")
        print("[load_model] Model not found, attempting download")
        try:
            download_file_from_google_drive(FILE_ID, MODEL_PATH)
            print("[load_model] Download succeeded")
            st.success("✅ Model downloaded.")
        except Exception as e:
            print("[load_model] Download failed:", e)
            st.error("Failed to download model. See logs.")
            # Stop the app to avoid calling load_model on missing file
            st.stop()

    # Final verification
    if not os.path.exists(MODEL_PATH):
        print("[load_model] Model path still missing after download attempts")
        st.error("Model file missing after download. Check logs.")
        st.stop()

    size = os.path.getsize(MODEL_PATH)
    print(f"[load_model] Loading model from {MODEL_PATH} (size={size} bytes)")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("[load_model] Model loaded successfully")
        return model
    except Exception as e:
        print("[load_model] Error loading model:", e)
        raise

model = load_model()

# Load disease info
plant_disease = {}
try:
    with open("plant_disease.json", "r") as file:
        plant_disease = json.load(file)
except Exception as e:
    print("plant_disease.json not found or invalid:", e)

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

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((160, 160))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_disease(image):
    img = preprocess_image(image)
    prediction = model.predict(img)
    index = int(np.argmax(prediction))
    return plant_disease.get(str(index), labels[index])

st.title("🌱 Plant Disease Detection")
st.write("Upload a leaf image to detect plant disease")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict Disease"):
        with st.spinner("Analyzing image..."):
            try:
                result = predict_disease(image)
                st.success("Prediction Complete")
                st.subheader("🦠 Disease Information")
                st.write(result)
            except Exception as e:
                st.error("Prediction failed — see logs.")
                print("Prediction error:", e)
