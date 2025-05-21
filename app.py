import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import gdown

# -------------------- Configurations --------------------
IMG_SIZE = (600, 600)
MODEL_PATH = "best_model.h5"
FILE_ID = "10gGgSNo9BZaOTlY2ewYTXcbLaSJP9GkW"
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"
CLASS_NAMES = ['Normal', 'Pneumonia-Bacterial', 'Viral Pneumonia']
LAST_CONV_LAYER = 'block5_conv3'

# -------------------- Page Layout --------------------
st.set_page_config(page_title="🫁 Chest X-ray Classifier", layout="centered")
st.markdown("<h1 style='text-align:center; color:#2E86C1;'>🫁 Chest X-ray Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#555;'>Upload a chest X-ray image and the model will classify it and visualize its decision using Grad-CAM.</p>", unsafe_allow_html=True)
st.markdown("<hr style='border: 2px solid #2E86C1;'>", unsafe_allow_html=True)

# -------------------- Load Model --------------------
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH, compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# -------------------- Upload Image --------------------
uploaded_file = st.file_uploader("📤 Upload a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        # Check if image is grayscale or has low variation (indicating possible X-ray)
        if image is None or image.std() < 5:
            st.error("❌ This does not appear to be a valid image.")
        elif image.shape[0] < 100 or image.shape[1] < 100:
            st.warning("⚠️ Image resolution is too low.")
        else:
            original_shape = image.shape
            image_resized = cv2.resize(image, IMG_SIZE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image_eq = clahe.apply(image_resized)
            image_rgb = cv2.merge([image_eq, image_eq, image_eq])
            image_norm = image_rgb.astype('float32') / 255.0
            img_array = np.expand_dims(image_norm, axis=0)

            preds = model.predict(img_array)
            class_idx = int(np.argmax(preds[0]))
            confidence = float(np.max(preds[0]))

            if confidence < 0.60:
                st.warning("⚠️ This image does not appear to be a valid chest X-ray.")
            else:
                predicted_class = CLASS_NAMES[class_idx]
                st.success(f"✅ **Prediction:** {predicted_class} ({confidence * 100:.2f}%)")
                st.image(image_rgb, caption="🩻 Uploaded Image", use_column_width=True)

                # -------------------- Grad-CAM --------------------
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
                overlay = cv2.addWeighted(image_rgb, 0.6, heatmap, 0.4, 0)

                st.markdown("<hr style='border: 1px solid #aaa;'>", unsafe_allow_html=True)
                st.subheader("🔍 Grad-CAM Heatmap")
                st.image(overlay, caption="🔥 Model Explanation", use_column_width=True)

    except Exception as e:
        st.error(f"❌ Failed to process the image.\n\n**{str(e)}**")
