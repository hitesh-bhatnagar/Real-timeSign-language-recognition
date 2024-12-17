import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('asl_model.h5')
labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]


st.title("ASL Sign Language Recognition")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    img_resized = cv2.resize(img, (64, 64)) / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)
    prediction = model.predict(img_resized)
    label = labels[np.argmax(prediction)]

    st.image(img, caption=f"Prediction: {label}", use_column_width=True)

    st.markdown(
        f"<h2 style='text-align: center; color: green;'>Prediction: {label}</h2>",
        unsafe_allow_html=True
    )