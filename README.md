# 🐱🐶 Cat or Dog Recognizer 🖼️  

![Cat or Dog Image](https://github.com/user-attachments/assets/b2bced6d-b0e7-4ecb-89c1-29f58223c00b)

## Overview

The **Cat or Dog Recognizer** is a machine learning-based image classifier designed to distinguish between images of cats and dogs. Leveraging a Convolutional Neural Network (CNN), it accurately classifies images based on patterns learned from a training dataset. This project demonstrates the use of deep learning techniques to solve a fundamental computer vision problem.

## Live Demo

Try out the Cat or Dog Recognizer! 👉🏻 [![Experience It! 🌟](https://img.shields.io/badge/Experience%20It!-blue)](https://paws-or-claws-predictor.streamlit.app/)

<br>

_Below is a preview of the Cat or Dog Recognizer in action. Upload an image to see the prediction! 👇🏻_

<p align="center">
  <img src="https://github.com/user-attachments/assets/a46cda1c-b8fc-43cc-b65e-cf78cefd3c13" alt="cat_or_dog_recognition">
</p>

<br>

## Table of Contents

1. [Features](#features)
2. [Dataset](#dataset)
3. [Model Training](#model-training)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Technologies Used](#technologies-used)
7. [Results](#results)
8. [Conclusion](#conclusion)
9. [Future Enhancements](#future-enhancements)
10. [License](#license)
11. [Contact](#contact)

<br>

## Features🌟

- **Real-time Image Classification**: Upload an image to instantly classify it as a cat or a dog.
- **Interactive Interface**: Built using Streamlit for a smooth user experience.
- **Model Accuracy**: High accuracy achieved through a trained deep learning model.
- **Optimized for quick processing**, providing a fast classification response even for large datasets.

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

### Data Processing:
- **Preprocessing**: The images are resized to a uniform dimension for consistency, normalized to improve model convergence, and split into training and testing sets to evaluate model performance.

This dataset provides a solid foundation for training image classification models to distinguish between cats and dogs, making it ideal for deep learning-based image recognition tasks.

<br>

## Model Training🧠

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

1. **Clone the repository**:
   ```bash
   https://github.com/hk-kumawat/Cat-or-Dog-Recognizer.git
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

<br>

## Usage🚀

1. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

2. **Upload an Image**: Choose an image of a cat or dog to classify.
3. **Results**: The app will display the classification result, showing whether the image is of dog or cat.
<br>

## Technologies Used💻

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

The **Cat or Dog Recognizer** model demonstrates excellent accuracy in classifying images. Below is an illustrative diagram that showcases the **basic workflow** of the model's operation, where the **Convolutional Neural Network (CNN)** analyzes images to classify them as either a **Cat** or a **Dog**.

<p align="center">
  <img src="https://github.com/user-attachments/assets/6139df4f-1370-4d36-9225-c9433cd41c50" alt="Cat and Dog CNN workflow" width="600"/>
</p>

In the example above, the CNN processes the image, labels it as a **Dog** or **Cat**, and outputs the classification result.

<br>

## Conclusion📚

The **Cat or Dog Recognizer** project showcases the power of deep learning and image classification in solving real-world problems. By leveraging a Convolutional Neural Network (CNN), the model effectively classifies images, offering a fun and engaging way to explore computer vision and machine learning concepts. 

This project serves as a solid foundation for further exploration into image classification, and future enhancements could include adding support for additional animal classifications or improving model accuracy with more advanced techniques. It exemplifies the exciting potential of machine learning in practical applications and encourages further experimentation and learning in the field.


<br>


## Future Enhancements🚀

1. **Support for More Animal Classifications**: Expand the model to recognize other animals, such as birds, horses, etc., using multi-class classification.

2. **Improved Model Accuracy**: Enhance the model's accuracy with a larger dataset, data augmentation, and pre-trained models (e.g., ResNet, InceptionV3).

3. **Better Classification Results Display**: Show confidence scores, additional animal information, and utilize image segmentation for better transparency.

4. **Real-time Webcam Classification**: Allow users to classify animals directly from their webcam in real-time.

5. **Integration with Other Platforms**: Enable users to classify images from social media platforms and easily share results.

6. **Explainability**: Use tools like **Grad-CAM** or **LIME** to visually explain the model’s predictions.


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

---

## Thanks for checking out this cute classifier! 🐱🐶

> "A dog is a loyal companion, and a cat is a mysterious friend." - Anonymous

