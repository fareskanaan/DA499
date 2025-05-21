import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import gdown
from PIL import Image

# -------------------- Settings --------------------
IMG_SIZE = (512, 512)
MODEL_PATH = "effnet_b4_model.h5"
FILE_ID = "10gGgSNo9BZaOTlY2ewYTXcbLaSJP9GkW"
CLASS_NAMES = ['Normal', 'Pneumonia-Bacterial', 'Viral Pneumonia']

# -------------------- App Interface --------------------
st.set_page_config(page_title="🫁 Chest X-ray Classifier", layout="centered")
st.title("🫁 AI Chest X-ray Classifier")
st.markdown("Upload a chest X-ray image for classification")

# -------------------- Model Loading with Validation --------------------
@st.cache_resource
def load_ai_model():
    try:
        if not os.path.exists(MODEL_PATH):
            with st.spinner('Downloading model...'):
                gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
        
        # Verify model file integrity
        if os.path.getsize(MODEL_PATH) < 1024*1024:  # Check if file is too small
            raise ValueError("Model file appears corrupted (too small)")
            
        model = load_model(MODEL_PATH, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

try:
    model = load_ai_model()
except Exception as e:
    st.error(f"Critical error loading model: {str(e)}")
    st.stop()

# -------------------- Image Processing --------------------
def preprocess_image(image):
    try:
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
        image = cv2.resize(image, IMG_SIZE)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        image = cv2.merge([image, image, image])
        return image.astype('float32') / 255.0
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        st.stop()

# -------------------- Main App --------------------
uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Read image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess and predict
        processed_img = preprocess_image(image)
        img_array = np.expand_dims(processed_img, axis=0)
        
        with st.spinner('Analyzing...'):
            preds = model.predict(img_array, verbose=0)
            
        class_idx = np.argmax(preds[0])
        predicted_class = CLASS_NAMES[class_idx]
        confidence = np.max(preds[0]) * 100
        
        st.success(f"**Result:** {predicted_class} ({confidence:.1f}% confidence)")
        
        # Show probabilities
        st.subheader("Detailed Probabilities:")
        for i, (cls, prob) in enumerate(zip(CLASS_NAMES, preds[0])):
            st.progress(float(prob), text=f"{cls}: {prob*100:.1f}%")
            
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
