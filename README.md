<a id="readme-top"></a>

# 🐱🐶 Cat or Dog Recognizer 🖼️  

![Cat or Dog Image](https://github.com/user-attachments/assets/b2bced6d-b0e7-4ecb-89c1-29f58223c00b)

## Overview

This project is an image classification system built using **Convolutional Neural Networks (CNN)** and **Streamlit**. It allows users to upload images or select from sample images to identify whether the image contains a cat or a dog. The application features a user-friendly interface, real-time predictions, and interactive elements like balloons animation on successful prediction. The model is trained on a large dataset of cat and dog images and achieves high accuracy in distinguishing between the two pets.

## Live Demo

Try out the Cat or Dog Recognizer! 👉🏻 [![Experience It! 🌟](https://img.shields.io/badge/Experience%20It!-blue)](https://paws-or-claws-predictor.streamlit.app/)

<br>

_Below is a preview of the app in action. Upload an image or select a sample to see the prediction! 👇🏻_

<p align="center">
  <img src="https://github.com/user-attachments/assets/e7451bd3-63a3-4da5-9cd4-d23d75ab10cd">
</p>

<br>

## Learning Journey

I developed this project to explore deep learning and computer vision while creating something fun and practical. Here’s a snapshot of my journey:

- **Inspiration:**  
  As a pet lover and AI enthusiast, I wanted to create a practical application that combines both interests. The challenge of distinguishing between cats and dogs, which can be tricky even for humans in some cases, seemed like a perfect project to tackle with deep learning.

- **Why I Made It:**  
  I aimed to build a system that could accurately classify pet images while learning about CNNs, image processing, and web deployment. This project served as a hands-on way to understand the complete machine learning pipeline from data preparation to model deployment.

- **Challenges Faced:**  
  - **Data Preprocessing:** Handling corrupt images and ensuring consistent input formats was a significant challenge.
  - **Model Architecture:** Finding the right balance of layers and parameters to prevent overfitting while maintaining accuracy.
  - **Image Augmentation:** Implementing effective augmentation techniques to improve model generalization.
  - **Web Integration:** Creating a seamless user experience with Streamlit while handling various image formats and sizes.

- **What I Learned:**  
  - **Deep Learning:** Practical experience with CNN architecture and training.
  - **Image Processing:** Techniques for handling and preprocessing image data.
  - **Data Augmentation:** Methods to increase dataset diversity and improve model robustness.
  - **Web Development:** Building an interactive web interface using Streamlit.

<br>

## Table of Contents

1. [Features](#features)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Training](#model-training)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Technologies Used](#technologies-used)
8. [Results](#results)
9. [Directory Structure](#directory-structure)
10. [Contributing](#contributing)
11. [License](#license)
12. [Contact](#contact)
  
<br>


## Features🌟

- **Accurate Classification:**  
  Utilizes a CNN to reliably differentiate between cats and dogs.

- **Interactive Interface:**  
  Upload your own image or select from sample images for quick predictions.

- **Real-Time Predictions:**  
  Get immediate results along with confidence scores.

- **Responsive Design:**  
  Seamless experience on both desktop and mobile devices.
  
<br>

## Dataset📊

The **Dogs vs. Cats** dataset is a widely used computer vision dataset for classifying images as either containing a dog or a cat. Developed in partnership between **Petfinder.com** and **Microsoft**, this dataset is a subset of a much larger collection of 3 million manually annotated images.

### Dataset Details:
- **Download Size**: 824 MB
- **Total Images**: 24,961 images 
  - **Cat Images**: 12,491
  - **Dog Images**: 12,470

<br>

The dataset is organized into the following structure:

```bash
kagglecatsanddogs_3367a
 ├── readme[1].txt
 ├── MSR-LA - 3467.docx
 └── PetImages
     ├── Cat    (Contains 12,491 images)
     └── Dog    (Contains 12,470 images)
```

### Source:
- **[Kaggle Dogs vs Cats Dataset](https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset)**: This is the dataset used for training the model, containing labeled images of cats and dogs.

<br>

## 🔄Data Preprocessing:

The notebook performs several steps to prepare the data:

1. **Loading Libraries & Dataset:**  
   - Uses libraries such as pandas, numpy, matplotlib, and Keras for image processing.
   - Iterates through the dataset folders to collect image paths and assign labels (0 for cats, 1 for dogs).

2. **Creating a DataFrame:**  
   - Combines image file paths and labels into a shuffled DataFrame for organized processing.

3. **Cleaning Data:**  
   - Identifies and removes invalid or non-image files (e.g., system files like Thumbs.db).

4. **Visualization & Class Distribution:**  
   - Displays grids of sample images and uses count plots to verify class balance.


<br>

## 🧠Model Training

The **Cat or Dog Classifier** project uses a **Convolutional Neural Network (CNN)** to classify images as either a cat or a dog. The model is trained with techniques aimed at improving its accuracy and robustness.

### Key Aspects of Model Training:

- **Model Architecture**: A Convolutional Neural Network (CNN) is used for image classification, which is effective in capturing spatial hierarchies in images.
  
- **Data Augmentation**: To enhance the model’s generalization and prevent overfitting, data augmentation techniques such as:
  - **Flipping** (horizontal and vertical)
  - **Zooming** (random zoom in/out)
  - **Rotating** (random rotations)
  
  These techniques help create variations of the original dataset, increasing model robustness.

- **Evaluation**: The model is evaluated based on its accuracy, with additional attention given to **minimizing overfitting** through validation and test splits.

### Final Model Artifacts:
- **`cat_dog_model.h5`**: The trained model saved as an H5 file, which can be used to predict whether an uploaded image contains a cat or a dog.


<br>

## Installation🛠


1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/hk-kumawat-cat-or-dog-recognizer.git
   cd hk-kumawat-cat-or-dog-recognizer
   ```

2. **Create & Activate a Virtual Environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate       # On Windows: venv\Scripts\activate
   ```

3. **Install Required Packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application:**
   ```bash
   streamlit run app.py
   ```
   
<br>

## Usage🚀

1. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

2. **Choose Input Method:**
   - Upload your own image
   - Select from sample images

3. **View Results:**
   - See the uploaded/selected image
   - Get instant prediction
   - Enjoy the celebration animation!

### Using the Jupyter Notebook

To explore the model development process:
1. **Open the Notebook:**
   ```bash
   jupyter notebook "Dog_vs_Cat_Image_Classification.ipynb"
   ```
2. **Execute the cells** to follow the complete model development pipeline.

<br>

## 💻Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - `tensorflow`
  - `keras`
  - `numpy`
  - `opencv-python`
  - `streamlit`
  - `matplotlib` (for visualizations)
- **Modeling Framework**: Keras (for CNN model)

<br>


## Results🏆

### Model Performance

- **Training Accuracy:** ~95%
- **Validation Accuracy:** ~93%
- **Features:**
  - Fast inference time (<1s)
  - Robust to various image sizes
  - Handles different image formats

### System Performance

- **Average Response Time:** <2s
- **Supported Image Formats:** JPG, JPEG, PNG
- **Memory Usage:** ~200MB
- **Concurrent Users:** 50+


<br>

## 📁Directory Structure

```plaintext
hk-kumawat-cat-or-dog-recognizer/
├── README.md                           # Project documentation
├── Dog_vs_Cat_Image_Classification.ipynb  # Jupyter Notebook for training and evaluation
├── LICENSE                             # License information
├── app.py                              # Streamlit application for image classification
├── requirements.txt                    # List of dependencies
├── Sample Images of Cat and Dog/        # Folder containing sample images
│   ├── Cats/                          # Sample images of cats
│   └── Dogs/                          # Sample images of dogs
└── model/                              # Saved models and related files
    └── cat_dog_model.h5                # Trained CNN model
```

<br>

## 🤝Contributing

Contributions make the open source community such an amazing place to learn, inspire, and create. 🙌 Any contributions you make are greatly appreciated! 😊

Have an idea to improve this project? Go ahead and fork the repo to create a pull request, or open an issue with the tag **"enhancement"**. Don't forget to give the project a star! ⭐ Thanks again! 🙏

<br>

1. **Fork** the repository.

2. **Create** a new branch:
   ```bash
   git checkout -b feature/YourFeatureName
   ```

3. **Commit** your changes with a descriptive message.

4. **Push** to your branch:
   ```bash
   git push origin feature/YourFeatureName
   ```

5. **Open** a Pull Request detailing your enhancements or bug fixes.


<br>


## License📝

This project is licensed under the **MIT License** - see the [LICENSE](./LICENSE) file for details.

<br>

## Contact

### 📬 Get in Touch!
I’d love to connect and discuss further:

- [![GitHub](https://img.shields.io/badge/GitHub-hk--kumawat-blue?logo=github)](https://github.com/hk-kumawat) 💻 — Explore my projects and contributions.
- [![LinkedIn](https://img.shields.io/badge/LinkedIn-Harshal%20Kumawat-blue?logo=linkedin)](https://www.linkedin.com/in/harshal-kumawat/) 🌐 — Let’s connect professionally.
- [![Email](https://img.shields.io/badge/Email-harshalkumawat100@gmail.com-blue?logo=gmail)](mailto:harshalkumawat100@gmail.com) 📧 — Send me an email for discussions and queries.

<br>

## Thanks for checking out the Cat or Dog Recognizer! 🐱🐶

> "Every pet deserves to be recognized, even by AI!" - Anonymous

<p align="right">(<a href="#readme-top">back to top</a>)</p>
