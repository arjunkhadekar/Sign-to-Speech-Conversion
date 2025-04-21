#  Real-Time American Sign Language (ASL) Recognition using CNNs
A high-accuracy, real-time ASL recognition system powered by TensorFlow and OpenCV.

---

##  Overview

This project implements a **real-time ASL recognition system** that translates static hand gestures (A–Z) into text. The system is built using **Convolutional Neural Networks (CNNs)** and is capable of smooth real-time prediction on standard hardware with a test accuracy of **98.77%**.

---

## Features

-  Real-time hand gesture recognition using webcam
-  High-accuracy CNN trained on ~87,000 labeled images
-  Fast inference (~24 FPS)
-  Live prediction display with GUI
-  Temporal smoothing for stable predictions
-  Adaptive ROI and background subtraction

## Model Architecture
- Input: 64×64×1 grayscale image
- Conv Block 1: 32 filters → ReLU → MaxPooling → BatchNorm
- Conv Block 2: 64 filters → ReLU → MaxPooling → BatchNorm
- Conv Block 3: 128 filters → ReLU → MaxPooling → BatchNorm
- Flatten → Dense (512) → Dropout (0.5) → Output (26 classes, Softmax)



