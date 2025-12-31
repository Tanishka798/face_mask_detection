import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("mask_detector.keras")

# Class labels (ORDER MATTERS)
labels = ["Mask", "No Mask"]

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0]
    class_id = np.argmax(prediction)
    confidence = prediction[class_id] * 100

    print(f"Image: {image_path}")
    print(f"Prediction: {labels[class_id]}")
    print(f"Confidence: {confidence:.2f}%")
    print("-" * 40)

# Test with multiple images
predict_image("test_images/test1.jpg")
predict_image("test_images/test2.jpg")
predict_image("test_images/test3.jpg")