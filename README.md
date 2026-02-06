# Emotion Detection from Audio (RAVDESS)

## My Information
* **Name:** P.Mayuk Jain
* **ID:** 2025A7CS0149P

##  Project Description
This project implements a Deep Learning model (2D CNN) to classify human emotions from audio speech. Using the **RAVDESS** dataset, the model detects 8 distinct emotions: *Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, and Surprised*.

The system processes raw audio waveforms into **Log-Mel Spectrograms** and uses a Convolutional Neural Network trained with advanced regularization techniques to achieve high accuracy and generalization.

##  Key Features
* **Robust Preprocessing:** Silence trimming and uniform audio padding (3 seconds).
* **Data Augmentation:** Implemented **Noise Injection** and **Pitch Shifting** to prevent the model from memorizing specific voice actors.
* **Model Architecture:** * 3 Convolutional Blocks (Conv2D + BatchNorm + MaxPool).
  * **Global Average Pooling (GAP)** instead of Flattening to reduce overfitting.
  * Dropout layers (0.1 and 0.3) for regularization.
* **Bias Check:** evaluated performance on Male vs. Female speakers to ensure no "Pitch Bias" exists.

##  Model Performance
The model was evaluated on a strictly unseen test set (stratified 10% split).

| Metric | Score |
| :--- | :--- |
| **Overall Accuracy** | **98%** |
| **Macro F1-Score** | **0.9801** |
| **Male Speaker Accuracy** | 98% |
| **Female Speaker Accuracy** |
