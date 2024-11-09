# streamlit_app.py

import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Load the pre-trained model
model = load_model('model/cat_dog_model.h5')

# Define a function for image classification
def classify_image(image_path, model):
    img = load_img(image_path, target_size=(128, 128))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return '🐶 Dog' if prediction[0] > 0.5 else '🐱 Cat'

# Streamlit UI
st.title("🐾 Cat vs Dog Classifier 🐾")
st.write("Upload an image below, and let our AI tell you if it's a **Cat** or a **Dog**!")

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Save the image for prediction
    image_path = "uploaded_image.jpg"
    image.save(image_path)
    
    # Predict button
    if st.button("Predict"):
        with st.spinner('Predicting...'):
            label = classify_image(image_path, model)
        st.success(f"Prediction complete! 🎉 It's a {label}!")
        st.balloons()

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>💡 Brought to Life By - <strong>Harshal Kumawat</strong></p>", unsafe_allow_html=True)
