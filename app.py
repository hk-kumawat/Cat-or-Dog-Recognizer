import streamlit as st
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image


# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Cat🐱 & Dog🐶 Identifier",
    page_icon="🐾",
    layout="wide"
)

# Load the pre-trained model
model = load_model('model/cat_dog_model.h5')

# Classification function
def classify_image(image, model):
    img = image.resize((128, 128))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0][0]
    return (
        'Meow!🐾 It’s a Cat!🐱 Look at that cute face!'
        if prediction < 0.5
        else 'Bark!🐶 Your furry friend just got identified!'
    )

# Streamlit UI
st.title("Dog 🐶 & Cat 🐱 Identifier 🐾")
st.write(
    "Upload an image **or** select a sample image below, and let our AI tell you if it's a **Cat** or a **Dog**!"
)

# Create two tabs: one for file upload and one for sample image selection
tab_upload, tab_sample = st.tabs(["Upload Image", "Select Sample Image"])

with tab_upload:
    # File upload option
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Predict", key="upload_predict"):
            with st.spinner('Predicting...'):
                label = classify_image(image, model)
            st.success(label)
            st.balloons()

with tab_sample:
    # Define the folder that holds sample images
    sample_base_folder = "Sample Images of Cat and Dog"
    
    # Select category: Cats or Dogs
    category = st.selectbox("Select Category", ["Cats", "Dogs"])
    sample_folder = os.path.join(sample_base_folder, category)
    
    # Get list of image files from the selected folder
    image_files = [
        file for file in os.listdir(sample_folder)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    
    if image_files:
        # Let the user choose a sample image
        image_choice = st.selectbox("Select an Image", image_files)
        sample_image_path = os.path.join(sample_folder, image_choice)
        image = Image.open(sample_image_path)
        st.image(image, caption=f"Sample {category} Image", use_container_width=True)
        
        if st.button("Predict", key="sample_predict"):
            with st.spinner('Predicting...'):
                label = classify_image(image, model)
            st.success(label)
            st.balloons()
    else:
        st.warning(f"No sample images found in the folder: {sample_folder}")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center;'>💡 Brought to Life By - <strong>Harshal Kumawat</strong></p>",
    unsafe_allow_html=True
)
