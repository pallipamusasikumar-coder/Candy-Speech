# Candy Speech Detection

## Project Overview
Candy Speech Detection is a Deep Learning project that classifies audio into two categories:
- Speech
- Non-Speech (Background Noise)

The system extracts MFCC (Mel Frequency Cepstral Coefficients) features from audio files and trains a neural network to distinguish speech from background sounds.

## Project Structure

Candy Speech/
│
├── dataset/
│   ├── speech/
│   └── non_speech/
│
├── preprocessing/
│   └── audio_loader.py
│
├── models/
├── training/
├── realtime/
├── results/
├── app.py
├── requirements.txt
└── README.md

## Requirements

- Python 3.11
- TensorFlow
- NumPy
- Librosa
- Scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt