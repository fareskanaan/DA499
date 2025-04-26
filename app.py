import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import gdown

# -------------------- إعدادات --------------------
IMG_SIZE = (600, 600)
MODEL_PATH = "vgg16_stream.h5"
FILE_ID = "1zMEAPzM2QsUP_aTuJ6jBbO8yNpemlq4-"
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"
CLASS_NAMES = ['Normal', 'Pneumonia-Bacterial', 'Viral Pneumonia']
LAST_CONV_LAYER = 'block5_conv3'

# -------------------- واجهة الموقع --------------------
st.set_page_config(page_title="🫁 Chest X-ray Classifier", layout="centered")

st.markdown("<h1 style='text-align:center; color:#2E86C1;'>🫁 Chest X-ray Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#555;'>Upload a chest X-ray image. The model will classify and explain the result using Grad-CAM.</p>", unsafe_allow_html=True)
st.markdown("<hr style='border: 2px solid #2E86C1;'>", unsafe_allow_html=True)

# -------------------- تحميل النموذج --------------------
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH, compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# -------------------- رفع صورة --------------------
uploaded_file = st.file_uploader("📤 Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # المعالجة
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        original_shape = image.shape

        if original_shape[:2] != IMG_SIZE:
            st.info(f"🔄 Resized from original shape {original_shape} to {IMG_SIZE}")

        image = cv2.resize(image, IMG_SIZE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        image_rgb = cv2.merge([image, image, image])
        image_norm = image_rgb.astype('float32') / 255.0
        img_array = np.expand_dims(image_norm, axis=0)

        # تنبؤ النموذج
        preds = model.predict(img_array)
        class_idx = int(np.argmax(preds[0]))
        predicted_class = CLASS_NAMES[class_idx]
        confidence = np.max(preds[0]) * 100

        st.success(f"✅ **Prediction:** {predicted_class} ({confidence:.2f}%)")
        st.image(image_rgb, caption="🩻 Input Image", use_column_width=True)

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
        st.subheader("🔍 Model Explanation: Grad-CAM")
        st.image(overlay, caption="🔥 Grad-CAM Heatmap", use_column_width=True)

    except Exception as e:
        st.error(f"❌ Failed to process the image.\n\n**{str(e)}**")
