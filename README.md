---

# Sign Language Interpreter

This project is a real-time sign language interpreter that uses computer vision and deep learning techniques to convert hand gestures into text. It leverages **Convolutional Neural Networks (CNN)** and the **OpenCV** library for gesture recognition, along with **MediaPipe** for hand tracking.

## Project Overview

This interpreter allows for seamless communication by translating hand gestures from sign language into text. The system uses a camera to capture gestures and outputs the corresponding text, making it an accessible tool for sign language users.

### Features
- **Real-Time Gesture Recognition**: Detects and recognizes hand gestures from the camera feed in real-time.
- **Custom Gesture Training**: Allows users to add and train custom gestures for sign language recognition.
- **Text Output**: Converts recognized gestures into text and displays it on the screen.
- **User-Friendly**: Easy-to-use interface for both training and recognition modes.

## Technologies Used
- **Python**: Primary programming language for the implementation.
- **OpenCV**: For real-time image processing and camera interface.
- **TensorFlow/Keras**: For building and training the Convolutional Neural Network (CNN) for gesture recognition.
- **MediaPipe**: Used for hand tracking and extracting key features from the hand.
- **NumPy, pandas**: For data processing and manipulation.
- **scikit-learn**: For model evaluation and classification.

## Requirements

1. Python 3.7 or higher
2. Install the required libraries by running the following command:

```bash
pip install -r requirements.txt
```

### Required Libraries
- `tensorflow`
- `keras`
- `opencv-python`
- `mediapipe`
- `numpy`
- `pandas`
- `scikit-learn`
- `os`

## How to Use

1. **Clone the repository:**

```bash
git clone https://github.com/AT8Cool/Sign_Language_interpretor.git
cd Sign_Language_interpretor
```

2. **Running the Interpreter:**

To start the interpreter, run the following:

```bash
python sign_language_interpreter.py
```

This will open the camera feed and begin recognizing gestures in real-time.

3. **Training Mode:**

   To train new gestures:
   - Press `T` to enter training mode.
   - Label the gesture (e.g., "Hello", "Thank You").
   - Perform the gesture, and the system will capture images to train on.
   - The program will automatically store the images and label them.

4. **Real-Time Recognition:**
   After training, press `R` to start the real-time gesture recognition. It will display the recognized gesture as text on the screen.

## Training Custom Gestures

1. Capture images of hand gestures that correspond to specific sign language signs.
2. Label each gesture for training.
3. The model will learn from these labeled images to recognize and translate them in real-time.

## Contributing

Feel free to fork, modify, and submit pull requests. If you add a new feature, please make sure to update the README and the training process.
I intend to enhance the accessibility of this project to reach more people.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
