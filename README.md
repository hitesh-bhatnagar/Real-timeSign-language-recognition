# Real-time-Sign-language-recognition

This project is a **Real-Time Sign Language Recognition System** that uses deep learning techniques to classify American Sign Language **(ASL)** gestures. It leverages **TensorFlow**, **MobileNetV2** (a pre-trained model), and **Streamlit** to provide real-time predictions through a webcam or image uploads.

### Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Tech Stack](#tech-stack)
4. [Directory Structure](#directory-structure)
5. [Installation and Usage](#installation-and-usage)
6. [Step-by-step Setup](#step-by-step-setup)
7. [Future Enhancements](#future-enhancements)
8. [Author](#author)

### Project Overview

This project aims to break communication barriers for the deaf and hard of hearing community by developing a system that can:
* Recognize static ASL gestures (A-Z, space, nothing, and delete).
* Perform **real-time predictions** through a webcam.
* Classify uploaded images of gestures.

#### Dataset
The model was trained using a [dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) containing images for ASL gestures.

### Features
1. **Real-Time Gesture Recognition:** Uses the webcam to predict ASL gestures in real time.
2. **Image Upload Prediction:** Upload images of hand gestures to get predictions instantly.
3. **Accurate Results:** Leveraged MobileNetV2 to enhance accuracy.
4. **Easy Deployment:** Web application using **Streamlit** for user-friendly interaction.

### Tech Stack

This project utilizes the following technologies:
* **Python:** Programming language.
* **TensorFlow/Keras:** Framework for deep learning model development.
* **MobileNetV2:** Pre-trained model for transfer learning.
* **OpenCV:** Real-time image processing.
* **Streamlit:** For building an interactive web application.
* **Google Colab:** Model training and testing environment with GPU support.

### Directory Structure

📦 Real-Time-Sign-Language-Recognition
```
│
├── asl_model.h5                          # Trained model file
├── Streamlit_app.py                      # Streamlit app for image uploads and predictions
├── Real-time-sign-recognition_asl_model.py   # Real-time prediction using webcam
├── Real_Time_Sign_language_Recognition.ipynb # Colab notebook for model training
├── MobileNetV2_real_time_sign_recognition.ipynb # Pre-trained MobileNetV2 model implementation
└── README.md                             # Project documentation
```

### Installation and Usage

#### Pre-requisites
1. Python 3.x
2. TensorFlow
3. OpenCV
4. Streamlit

#### Step-by-step Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/YourGitHubUsername/Real-Time-Sign-Language-Recognition.git
   cd Real-Time-Sign-Language-Recognition
   ```
2. Install dependencies:
   ```bash
   pip install tensorflow opencv-python-headless streamlit numpy matplotlib pillow
   ```
3. Run the Streamlit app (Image Upload Prediction):
   ```bash
   streamlit run Streamlit_app.py
   ```
4. Run real-time webcam gesture recognition:
   ```bash
   python Real-time-sign-recognition_asl_model.py
   ```

### Future Enhancements

1. **Dynamic Gesture Recognition:** Recognize hand movements to predict words or sentences.
2. **Speech Recognition:** Convert recognized gestures into spoken words using text-to-speech.
3. **Mobile Application:** Deploy the model on mobile devices for accessibility.
4. **Extended Dataset:** Enhance accuracy with a larger and more diverse dataset.

## Author

### Hitesh Bhatnagar

* [GitHub](https://github.com/hitesh-bhatnagar)
* [LinkedIn](https://www.linkedin.com/in/hitesh-bhatnagar-5a3b391ba)

