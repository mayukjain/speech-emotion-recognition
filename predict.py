import argparse
import numpy as np
import librosa
import tensorflow as tf
import os

SAMPLE_RATE = 22050
DURATION = 3.0
N_MELS = 128
TARGET_LENGTH = int(SAMPLE_RATE * DURATION)
MODEL_PATH = 'emotion_model.keras'
CLASSES_PATH = 'classes.npy'

def clean_audio(y):
    y_trimmed, _ = librosa.effects.trim(y, top_db=30)
    if len(y_trimmed) > TARGET_LENGTH:
        start = (len(y_trimmed) - TARGET_LENGTH) // 2
        return y_trimmed[start : start + TARGET_LENGTH]
    else:
        padding = TARGET_LENGTH - len(y_trimmed)
        return np.pad(y_trimmed, (0, padding), 'constant')

def get_spectrogram(y, sr):
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    return librosa.power_to_db(mels, ref=np.max)

def predict_emotion(audio_path):
    if not os.path.exists(audio_path):
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    classes = np.load(CLASSES_PATH)

    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    cleaned_y = clean_audio(y)
    spec = get_spectrogram(cleaned_y, SAMPLE_RATE)
    input_data = spec[np.newaxis, ..., np.newaxis] 

    predictions = model.predict(input_data, verbose=0)
    predicted_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) * 100
    predicted_label = classes[predicted_index]

    print(f"EMOTION: {predicted_label.upper()}")
    print(f"CONFIDENCE: {confidence:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    args = parser.parse_args()
    
    predict_emotion(args.file_path)
