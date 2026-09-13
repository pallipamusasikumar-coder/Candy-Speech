import os
import librosa
import numpy as np
from sklearn.metrics import accuracy_score
from joblib import load

model = load("speech_model.joblib")

SPEECH_PATH = "dataset/speech"
NON_SPEECH_PATH = "dataset/non_speech"

X = []
y = []

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

for file in os.listdir(SPEECH_PATH):
    if file.endswith(".wav"):
        X.append(extract_features(os.path.join(SPEECH_PATH, file)))
        y.append(1)

for file in os.listdir(NON_SPEECH_PATH):
    if file.endswith(".wav"):
        X.append(extract_features(os.path.join(NON_SPEECH_PATH, file)))
        y.append(0)

X = np.array(X)
y = np.array(y)

predictions = model.predict(X)

accuracy = accuracy_score(y, predictions)

print(f"Model Accuracy: {accuracy * 100:.2f}%")