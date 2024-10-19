# collect_images.py

import cv2
import os
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def get_hand_landmarks(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            return landmarks
    return None

def capture_hand_landmarks(label, num_samples=100, save_dir='dataset'):
    cap = cv2.VideoCapture(0)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    label_dir = os.path.join(save_dir, label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    print(f"Collecting {num_samples} samples for label: {label}")

    count = 0
    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = get_hand_landmarks(frame)
        if landmarks is not None:
            np.save(os.path.join(label_dir, f"{count}.npy"), landmarks)
            count += 1

        cv2.imshow("Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    label = input("Enter the label for sign (e.g., A, B, C, etc.): ")
    capture_hand_landmarks(label)
