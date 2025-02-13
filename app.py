import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image

# Load the pre-trained model
model = load_model('model/cat_dog_model.h5')

# Corrected classification function
def classify_image(image, model):
    img = image.resize((128, 128))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0][0] 
    
    return 'Meow!🐾 It’s a Cat!🐱 Look at that cute face!' if prediction < 0.5 else 'Bark!🐶 Your furry friend just got identified!'

# Streamlit UI
st.title("Dog 🐶 & Cat 🐱 Identifier 🐾")
st.write("Upload an image below, and let our AI tell you if it's a **Cat** or a **Dog**!")

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Predict"):
        with st.spinner('Predicting...'):
            label = classify_image(image, model)
        st.success(label)  
        st.balloons()

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>💡 Brought to Life By - <strong>Harshal Kumawat</strong></p>", unsafe_allow_html=True)
