import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import gdown

# -------------------- Configurations --------------------
IMG_SIZE = (600, 600)
MODEL_PATH = "vgg16_restored_final.h5"
FILE_ID = "10gGgSNo9BZaOTlY2ewYTXcbLaSJP9GkW"
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"
CLASS_NAMES = ['Normal', 'Pneumonia-Bacterial', 'Viral Pneumonia']
LAST_CONV_LAYER = 'block5_conv3'

# -------------------- Streamlit Page Setup --------------------
st.set_page_config(page_title="🫁 Chest X-ray Classifier", layout="centered")

st.markdown("<h1 style='text-align:center; color:#2E86C1;'>🫁 Chest X-ray Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#555;'>Upload a chest X-ray image. The model will classify and explain the result using Grad-CAM.</p>", unsafe_allow_html=True)
st.markdown("<hr style='border: 2px solid #2E86C1;'>", unsafe_allow_html=True)

# -------------------- Load Model --------------------
if not os.path.exists(MODEL_PATH):
    st.info("⏳ Downloading the model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

try:
    model = load_model(MODEL_PATH, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    st.success("✅ Model loaded successfully.")
except Exception as e:
    st.error(f"❌ Failed to load the model: {str(e)}")
    st.stop()

# -------------------- Chest X-ray Validation Function --------------------
def is_chest_xray(image):
    """
    Check if the image is likely a chest X-ray based on grayscale and intensity properties.
    Returns True if the image is a chest X-ray, False otherwise.
    """
    # Convert to RGB if not already
    if len(image.shape) == 2:
        image = cv2.merge([image, image, image])
    
    # Check if the image is near-grayscale (R, G, B channels are nearly identical)
    r, g, b = cv2.split(image)
    channel_diff = np.mean(np.abs(r - g)) + np.mean(np.abs(g - b)) + np.mean(np.abs(b - r))
    is_grayscale = channel_diff < 10  # Threshold for grayscale-like images
    
    # Check intensity distribution (chest X-rays have high contrast between bones and tissues)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    intensity_std = np.std(gray)
    is_high_contrast = intensity_std > 30  # Typical for chest X-rays
    
    return is_grayscale and is_high_contrast

# -------------------- Image Upload and Processing --------------------
uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Read and preprocess the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        original_shape = image.shape

        # Check if the image is a chest X-ray
        image_rgb = cv2.cvtColor(cv2.merge([image, image, image]), cv2.COLOR_BGR2RGB)
        if not is_chest_xray(image_rgb):
            st.error("❌ This is not a chest X-ray image. Please upload a valid chest X-ray.")
            st.stop()

        # Resize and apply CLAHE
        if original_shape[:2] != IMG_SIZE:
            st.info(f"🔄 Resized from original shape {original_shape} to {IMG_SIZE}")
        image = cv2.resize(image, IMG_SIZE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        image_rgb = cv2.merge([image, image, image])
        image_norm = image_rgb.astype('float32') / 255.0
        img_array = np.expand_dims(image_norm, axis=0)

        # Display the input image
        st.image(image_rgb, caption="🩻 Input Image", use_column_width=True)

        # Model prediction
        preds = model.predict(img_array)
        class_idx = int(np.argmax(preds[0]))
        predicted_class = CLASS_NAMES[class_idx]
        confidence = np.max(preds[0]) * 100

        st.success(f"✅ **Prediction:** {predicted_class} ({confidence:.2f}%)")

        # -------------------- Grad-CAM Visualization --------------------
        grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(LAST_CONV_LAYER).output, model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)[0]
        conv_outputs = conv_outputs[0]
        weights = tf.reduce_mean(grads, axis=(0, 1))
        cam = np.zeros(conv_outputs.shape[:2], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * conv_outputs[:, :, i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, IMG_SIZE)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        heatmap = (cam * 255).astype("uint8")
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # Convert to RGB for Streamlit
        overlay = cv2.addWeighted(image_rgb, 0.6, heatmap, 0.4, 0)

        st.markdown("<hr style='border: 1px solid #aaa;'>", unsafe_allow_html=True)
        st.subheader("🔍 Model Explanation: Grad-CAM")
        st.image(overlay, caption="🔥 Grad-CAM Heatmap (Highlights important regions)", use_column_width=True)

    except Exception as e:
        st.error(f"❌ Failed to process the image: {str(e)}")

# -------------------- Footer --------------------
st.markdown("<hr style='border: 1px solid #aaa;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#555;'>Built with Streamlit and TensorFlow. For chest X-ray classification.</p>", unsafe_allow_html=True)
