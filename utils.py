def preprocess_image(image):
    # Resize and normalize the image
    image = cv2.resize(image, (64, 64,3))
    image = image / 255.0
    return image
