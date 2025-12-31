# 😷 Face Mask Detection using Deep Learning

## 📌 Overview
This project is a Face Mask Detection system built using Deep Learning and Computer Vision.
It detects whether a person is wearing a face mask or not, using both static images and
real-time webcam input.

The model is trained using transfer learning with MobileNetV2 on a balanced dataset of over
7,500 images, achieving high validation accuracy while controlling overfitting using data
augmentation and early stopping.

---

## 🚀 Features
- Face mask detection on unseen images
- Real-time webcam detection
- Transfer learning using MobileNetV2
- Data augmentation for better generalization
- Early stopping to prevent overfitting
- Confidence score for predictions

---

## 🛠️ Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib

---

## 📂 Project Structure
```
face_mask_detection/
│
├── dataset/
│   ├── with_mask/
│   └── without_mask/
│
├── test_images/
│   ├── test1.jpg
│   ├── test2.jpg
│
├── train_mask_detector.py
├── detect_mask_video.py
├── predict_image.py
├── mask_detector.keras
└── README.md
```

---

## 📊 Dataset Information
- With Mask: 3,725 images
- Without Mask: 3,828 images
- Total images: ~7,500
- Balanced dataset suitable for binary classification

---

## 🧠 Model Workflow
1. Face detected using Haar Cascade Classifier
2. Image resized to 224×224
3. Pixel values normalized
4. MobileNetV2 extracts features
5. Dense layers classify Mask or No Mask

---

## ⚙️ How to Run the Project

### 1. Clone the Repository
git clone https://github.com/Tanishka798/face_mask_detection.git
cd face_mask_detection

### 2. Create and Activate Virtual Environment
python -m venv venv
venv\Scripts\activate

### 3. Install Dependencies
pip install tensorflow opencv-python numpy matplotlib

### 4. Train the Model
python train_mask_detector.py

The trained model is saved as:
mask_detector.keras

### 5. Test on Random Images
python predict_image.py

### 6. Run Real-Time Webcam Detection
python detect_mask_video.py
Press Q to exit the webcam window.

---

## 📈 Model Performance
- Training accuracy reached ~98%
- Validation accuracy peaked early, indicating fast convergence
- Early stopping was used to avoid overfitting

---

## 🧠 What I Learned
- Transfer learning using pre-trained CNNs
- Difference between training and validation accuracy
- Role of dropout and dense layers in regularization
- Image preprocessing and data augmentation
- Real-time inference using OpenCV

---

## 🔮 Future Improvements
- Deploy using Streamlit or Flask
- Convert model to TensorFlow Lite
- Improve face detection using deep learning-based detectors
- Add FPS counter and confidence visualization

---

## 👤 Author
Tanishka

---
