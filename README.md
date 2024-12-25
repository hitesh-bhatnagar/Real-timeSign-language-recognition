
---

# Real-Time Sign Language Recognition

This project is a **Real-Time Sign Language Recognition System** that utilizes deep learning to classify American Sign Language (**ASL**) gestures. It leverages **TensorFlow**, **MobileNetV2** (a pre-trained model), and **Streamlit** to provide real-time predictions via webcam or image uploads.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Tech Stack](#tech-stack)  
4. [Directory Structure](#directory-structure)  
5. [Installation and Usage](#installation-and-usage)  
6. [Step-by-Step Setup](#step-by-step-setup)  
7. [Future Enhancements](#future-enhancements)  
8. [Author](#author)  

---

## Project Overview

This project aims to break communication barriers for the deaf and hard-of-hearing community by developing a system that can:  
* Recognize static ASL gestures (A-Z, space, nothing, and delete).  
* Perform **real-time predictions** through a webcam.  
* Classify uploaded images of gestures.  

### Dataset
The model was trained using a publicly available [dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) containing labeled ASL gesture images.

---

## Features

1. **Real-Time Gesture Recognition:** Predict ASL gestures using the webcam in real time.  
2. **Image Upload Prediction:** Instantly classify hand gestures from uploaded images.  
3. **High Accuracy:** Utilizes the MobileNetV2 architecture for precise predictions.  
4. **User-Friendly Interface:** A simple and interactive web app powered by **Streamlit**.  

---

## Tech Stack

This project is built with the following technologies:  
* **Python:** The primary programming language.  
* **TensorFlow/Keras:** Framework for developing and training deep learning models.  
* **MobileNetV2:** Pre-trained model utilized for transfer learning.  
* **OpenCV:** For real-time image capture and processing.  
* **Streamlit:** Enables an interactive web application for predictions.  
* **Google Colab:** Model training and testing environment with GPU support.  

---

## Directory Structure

📦 Real-Time-Sign-Language-Recognition  
```
│
├── asl_model.h5                          # Trained model file
├── Streamlit_app.py                      # Web app for image uploads and predictions
├── Real-time-sign-recognition_asl_model.py   # Script for real-time webcam gesture recognition
├── Real_Time_Sign_Language_Recognition.ipynb # Colab notebook for model training
├── MobileNetV2_real_time_sign_recognition.ipynb # Implementation of MobileNetV2 for sign recognition
└── README.md                             # Project documentation
```

---

## Installation and Usage

### Prerequisites  
1. Python 3.x  
2. TensorFlow  
3. OpenCV  
4. Streamlit  

### Step-by-Step Setup  

1. Clone the repository:  
   ```bash
   git clone https://github.com/YourGitHubUsername/Real-Time-Sign-Language-Recognition.git
   cd Real-Time-Sign-Language-Recognition
   ```  
2. Install dependencies:  
   ```bash
   pip install tensorflow opencv-python-headless streamlit numpy matplotlib pillow
   ```  
3. Start the Streamlit app (for image uploads):  
   ```bash
   streamlit run Streamlit_app.py
   ```  
4. Launch real-time gesture recognition using webcam:  
   ```bash
   python Real-time-sign-recognition_asl_model.py
   ```  

---

## Future Enhancements  

1. **Dynamic Gesture Recognition:** Recognize sequences of gestures to predict full words or sentences.  
2. **Speech Integration:** Translate gestures into spoken words using text-to-speech tools.  
3. **Mobile Application:** Deploy the system as a mobile app for wider accessibility.  
4. **Enhanced Dataset:** Improve accuracy by training on larger and more diverse datasets.  

---

## Author  

### Hitesh Bhatnagar  

Connect with me:  
* [GitHub](https://github.com/hitesh-bhatnagar)  
* [LinkedIn](https://www.linkedin.com/in/hitesh-bhatnagar-5a3b391ba)  

---
