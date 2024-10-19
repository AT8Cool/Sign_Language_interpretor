import cv2
import numpy as np
from keras.models import load_model
import time

# Load the trained model
model = load_model('best_sign_language_model.keras')

# Dictionary to map model predictions to letters/signs
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
    17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

# Initialize webcam
cap = cv2.VideoCapture(0)

# Placeholder for storing recognized letters
sentence = ""


def predict_gesture(frame):
    # Preprocess the frame (resize and normalize)
    frame = cv2.resize(frame, (64, 64))  # Resize to match training image size (64x64)
    frame = np.expand_dims(frame, axis=0)  # Expand dimensions to fit model input
    frame = frame / 255.0  # Normalize

    # Predict the class (sign language gesture)
    prediction = model.predict(frame)

    # Print the prediction probabilities for all gestures
    print(f"Prediction Probabilities: {prediction}")

    pred_index = np.argmax(prediction)  # Get the index of the predicted class

    # Print the predicted gesture
    print(f"Predicted Gesture: {labels_dict.get(pred_index, '')}")

    return labels_dict.get(pred_index, '')  # Return the predicted gesture


last_prediction_time = time.time()  # Variable to track time of the last prediction

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Flip the frame if needed
    frame = cv2.flip(frame, 1)

    # Only predict a new gesture if 3 seconds have passed since the last one
    if time.time() - last_prediction_time >= 5:
        # Predict the gesture
        gesture = predict_gesture(frame)

        # If a valid gesture is predicted
        if gesture:
            sentence += gesture  # Append gesture to the sentence
            last_prediction_time = time.time()  # Update the time of the last prediction

    # Display the frame
    cv2.putText(frame, 'Press Q to exit', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Create a black rectangle at the bottom of the frame for the sentence box
    frame_height, frame_width, _ = frame.shape
    cv2.rectangle(frame, (0, frame_height - 50), (frame_width, frame_height), (0, 0, 0), -1)

    # Display the current sentence in the text box
    cv2.putText(frame, f'Sentence: {sentence}', (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                2)

    # Show the camera feed with the sentence
    cv2.imshow("Sign Language to Text", frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
