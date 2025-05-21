import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import gdown

# -------------------- Settings --------------------
IMG_SIZE = (512, 512)  # Updated to match your new model
MODEL_PATH = "effnet_b4_model.h5"
FILE_ID = "10gGgSNo9BZaOTlY2ewYTXcbLaSJP9GkW"  # Updated Google Drive file ID
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"
CLASS_NAMES = ['Normal', 'Pneumonia-Bacterial', 'Viral Pneumonia']
LAST_CONV_LAYER = 'top_conv'  # Updated for EfficientNet

# -------------------- App Interface --------------------
st.set_page_config(page_title="🫁 Chest X-ray Classifier", layout="centered")

st.markdown("<h1 style='text-align:center; color:#2E86C1;'>🫁 AI Chest X-ray Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#555;'>Upload a chest X-ray image for classification (Normal, Bacterial Pneumonia, or Viral Pneumonia)</p>", unsafe_allow_html=True)
st.markdown("<hr style='border: 2px solid #2E86C1;'>", unsafe_allow_html=True)

# -------------------- Model Loading --------------------
@st.cache_resource
def load_ai_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    model = load_model(MODEL_PATH, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = load_ai_model()

# -------------------- Image Processing --------------------
def preprocess_image(image):
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Enhanced preprocessing
    image = cv2.resize(image, IMG_SIZE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    image = cv2.merge([image, image, image])
    return image.astype('float32') / 255.0

def is_chest_xray(image):
    """Basic check if image appears to be a chest X-ray"""
    # Check if image is mostly dark with bright areas (like X-rays)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    return hist[0] > 0.3 * gray.size  # At least 30% dark pixels

# -------------------- Grad-CAM --------------------
def generate_gradcam(model, img_array, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)

    cam = cv2.resize(cam.numpy(), IMG_SIZE)
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam

# -------------------- Main App --------------------
uploaded_file = st.file_uploader("📤 Upload a chest X-ray image (JPG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Check if image appears to be a chest X-ray
        if not is_chest_xray(image):
            st.error("⚠️ This doesn't appear to be a chest X-ray image. Please upload a valid chest X-ray.")
            st.stop()
        
        # Display original image
        st.image(image, caption="🩻 Original Image", use_column_width=True)
        
        # Preprocess and predict
        processed_img = preprocess_image(image)
        img_array = np.expand_dims(processed_img, axis=0)
        
        preds = model.predict(img_array, verbose=0)
        class_idx = np.argmax(preds[0])
        predicted_class = CLASS_NAMES[class_idx]
        confidence = np.max(preds[0]) * 100
        
        # Display results
        st.success(f"**AI Diagnosis:** {predicted_class} (Confidence: {confidence:.1f}%)")
        
        # Generate Grad-CAM
        cam = generate_gradcam(model, img_array, LAST_CONV_LAYER)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        superimposed_img = cv2.addWeighted(
            cv2.cvtColor(np.uint8(255 * processed_img), cv2.COLOR_BGR2RGB), 
            0.6, 
            heatmap, 
            0.4, 
            0
        )
        
        st.subheader("🔍 AI Explanation (Grad-CAM)")
        st.image(superimposed_img, caption="Heatmap shows areas most important for diagnosis", use_column_width=True)
        
        # Show class probabilities
        st.subheader("📊 Prediction Probabilities")
        prob_data = {
            "Condition": CLASS_NAMES,
            "Probability": [f"{p*100:.1f}%" for p in preds[0]]
        }
        st.table(prob_data)
        
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")

# -------------------- Footer --------------------
st.markdown("---")
st.markdown("""
**How to use:**
1. Upload a frontal chest X-ray image
2. The AI will classify it as Normal, Bacterial Pneumonia, or Viral Pneumonia
3. See the heatmap showing which areas influenced the decision

*Note: This tool is for educational purposes only and not a substitute for professional medical advice.*
""")
