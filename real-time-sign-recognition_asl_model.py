import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('asl_model.h5')

# Define the class labels in the same order as your dataset
labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]

# Define image size for the model
IMG_SIZE = 64

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for prediction
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = np.expand_dims(img, axis=0) / 255.0

    # Predict the gesture
    predictions = model.predict(img)
    predicted_class = labels[np.argmax(predictions)]

    # Display the prediction on the screen
    cv2.putText(
        frame, f"Prediction: {predicted_class}", (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )
    cv2.imshow('ASL Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
