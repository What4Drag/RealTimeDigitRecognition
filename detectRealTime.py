import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('my_model.keras')

# Set up video capture from webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Preprocess frame (resize, convert to grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    resized = resized / 255.0  # Normalize pixel values
    
    # Reshape image for model input (add batch dimension)
    input_image = np.expand_dims(resized, axis=0)
    input_image = np.expand_dims(input_image, axis=-1)
    
    # Predict digit
    prediction = model.predict(input_image)
    digit = np.argmax(prediction)
    if np.max(prediction)>.5:
    # # Display predicted digit on frame
        cv2.putText(frame, str(digit), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, str(np.max(prediction)), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2)
    
    # Display the resulting frame
    cv2.imshow('Real-time Digit Recognition', frame)
    
    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

