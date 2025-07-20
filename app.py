import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------- SETUP --------------------------
st.set_page_config(
    page_title="Plant Leaf Health Detector",
    page_icon="🌿",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("keras_model.h5")

model = load_model()
class_names = ["Healthy 🌱", "Anomalous 🍂"]

# ---------------------- TITLE --------------------------
st.markdown("<h1 style='text-align: center; color: #2e7d32;'>🌿 Plant Leaf Health Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 16px;'>Upload a leaf image to detect plant health using AI.</p>", unsafe_allow_html=True)

# ---------------------- EXPLANATORY IMAGE --------------------------
st.image("pilot.jpg", caption="🧭 Healthy (left) vs Anomalous (right) Leaf", width=600)

st.markdown("---")

# ---------------------- IMAGE UPLOADER --------------------------
uploaded_file = st.file_uploader("📁 Upload a leaf image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"], help="Use a clear image with the leaf in focus")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Uploaded Image", use_container_width=True)

    st.markdown("🔄 Processing image...", unsafe_allow_html=True)

    # Preprocess
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)

    # Predict
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(predictions)) * 100

    # ---------------------- DISPLAY RESULT --------------------------
    st.markdown("---")
    st.markdown("<h3 style='text-align: center;'>🧠 AI Prediction Result</h3>", unsafe_allow_html=True)

    if predicted_index == 0:
        # ✅ Healthy
        st.success("✅ The plant leaf is predicted to be **Healthy**.")
        st.markdown(f"<p style='text-align:center; font-size:18px; color:green;'><strong>Confidence:</strong> {confidence:.2f}%</p>", unsafe_allow_html=True)
        st.progress(int(confidence))
    else:
        # ❌ Anomalous
        st.error("❌ The plant leaf shows **Anomalous (Diseased/Damaged)** characteristics.")
        st.markdown(f"<p style='text-align:center; font-size:18px; color:#b00020;'><strong>Confidence:</strong> {confidence:.2f}%</p>", unsafe_allow_html=True)
        st.progress(int(confidence))

    # ---------------------- BREAKDOWN --------------------------
    st.markdown("### 🔬 Confidence Breakdown")
    for i, prob in enumerate(predictions[0]):
        color = "green" if i == 0 else "#b00020"
        st.markdown(f"<p style='color:{color};'>{class_names[i]}: <strong>{prob * 100:.2f}%</strong></p>", unsafe_allow_html=True)

# ---------------------- SIDEBAR --------------------------
with st.sidebar:
    st.markdown("## ℹ️ About")
    st.info("""
This app uses a trained AI model to analyze plant leaves and detect whether they are **healthy** or **show signs of disease**.

🔧 Technologies:
- TensorFlow (Keras)
- Teachable Machine
- Streamlit

🧪 Model Input:
- Image resized to 224x224 pixels
- Pixel values normalized
    """)

# ---------------------- FOOTER --------------------------
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 13px;'>Made by Prakhar Pandey • Powered by TensorFlow & Streamlit</p>", unsafe_allow_html=True)
