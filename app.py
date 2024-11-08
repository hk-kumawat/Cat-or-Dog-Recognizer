import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from tensorflow.keras.models import model_from_json
import numpy as np
import os
import random

# =============================== Title ===============================
st.title("Cat 🐱 Or Dog 🐶 Recognizer")

# ============================= Title Image ============================
st.text("")
img_path_list = ["static/image_1.jpg", "static/image_2.jpg"]
index = random.choice([0, 1])
title_image = Image.open(img_path_list[index])
st.image(title_image, use_column_width=True)


# ======================== What It Will Predict =========================
st.write("## 3️⃣ What It Will Predict")
st.write("This app predicts if the uploaded image is of a Cat 🐈 or a Dog 🐕.")

# ===================== Upload and Display Image =======================
st.write("## 👁️‍🗨️ Time To See The Magic 🌀")
img_file_buffer = st.file_uploader("Upload an image here 👇🏻")

if img_file_buffer:
    try:
        uploaded_image = Image.open(img_file_buffer)
        st.write("Preview 👀 of Uploaded Image")
        st.image(uploaded_image, use_column_width=True)
    except Exception as e:
        st.write("### ❗ Invalid file or unable to display image. Please try again.")
else:
    st.write("### ❗ No image selected yet!")

# ============================== Predict ===============================
submit = st.button("👉🏼 Predict")

# ============================== Model =================================
@st.cache_resource
def load_model():
    try:
        with open("model/model.json", 'r') as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("model/model.h5")
        loaded_model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
        return loaded_model
    except Exception as e:
        st.write("### ❗ Error loading model. Ensure model files are correctly placed.")
        return None

model = load_model()

def process_image(image_path):
    IMG_SIZE = 50
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode="grayscale")
    img_array = img_to_array(img) / 255.0
    return img_array.reshape((1, 50, 50, 1))

def generate_result(prediction):
    st.write("## 🎯 RESULT")
    if prediction[0] < 0.5:
        st.write("## Model predicts it as an image of a CAT 🐱!")
    else:
        st.write("## Model predicts it as an image of a DOG 🐶!")

# ========================== Prediction Logic ==========================
if submit and img_file_buffer and model:
    try:
        if not os.path.exists("temp_dir"):
            os.makedirs("temp_dir")
        image_path = "temp_dir/test_image.png"
        uploaded_image.save(image_path)
        processed_image = process_image(image_path)

        st.write("👁️ Predicting...")
        prediction = model.predict(processed_image)
        generate_result(prediction)
    except Exception as e:
        st.write("### ❗ Oops... Something went wrong during prediction.")
else:
    if submit:
        st.write("### ❗ Please upload an image to predict.")

# ============================= Footer ================================
st.write("### ©️ Created By Harshal Kumawat")
