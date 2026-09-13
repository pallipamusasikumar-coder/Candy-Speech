# 🍬 Candy Speech Detection

**Candy Speech** is an intelligent audio classification system designed for real-time speech and sound detection using machine learning and audio signal processing.

The project extracts **MFCC (Mel-Frequency Cepstral Coefficients)** and other audio features from sound recordings and uses a trained machine-learning model to classify different types of audio.

## 🚀 Features

* 🎙️ Audio classification
* 🧠 Machine learning-based prediction
* 🎵 MFCC feature extraction
* 📊 Audio waveform and spectrogram visualization
* 📈 Dataset statistics and prediction probabilities
* ⚡ Real-time audio detection
* 💾 Trained model using Joblib
* 🖥️ Streamlit-based application interface
* 📁 Organized training, preprocessing, evaluation, and realtime modules

## 🧠 Classification

The current model supports classification of:

* **Human**
* **Animal**
* **Noise**

The system analyzes the characteristics of an audio signal and predicts the most likely class.

## 🛠️ Technologies Used

* **Python**
* **Streamlit**
* **Librosa**
* **NumPy**
* **Pandas**
* **Scikit-learn**
* **Matplotlib**
* **Joblib**
* **Audio Signal Processing**
* **Machine Learning**

## 📂 Project Structure

```text
Candy-Speech/
│
├── dataset/
│   ├── animal/
│   ├── human/
│   └── noise/
│
├── evaluation/
│   └── evaluate.py
│
├── preprocessing/
│   └── features.py
│
├── realtime/
│   ├── live_detection.py
│   └── predict.py
│
├── training/
│   └── train.py
│
├── app.py
├── dashboard.py
├── gui.py
├── sound_model.joblib
├── speech_model.joblib
├── requirements.txt
└── README.md
```

## 🔍 Feature Extraction

The system uses audio signal-processing techniques to extract meaningful information from sound files.

Important features include:

* MFCC
* Delta MFCC
* Delta-Delta MFCC
* Mean features
* Standard deviation features

These features are provided to the trained machine-learning model for classification.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/pallipamusasikumar-coder/Candy-Speech.git
```

### 2. Open the project

```bash
cd Candy-Speech
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## 📊 Model

The project includes trained machine-learning models:

```text
sound_model.joblib
speech_model.joblib
```

The main sound classification model uses extracted audio features to predict the category of an input audio file.

## 🔄 System Workflow

```text
Audio Input
     ↓
Audio Preprocessing
     ↓
Feature Extraction
     ↓
MFCC / Delta Features
     ↓
Trained ML Model
     ↓
Prediction
     ↓
Human / Animal / Noise
```

## 📈 Evaluation

The project contains an evaluation module for testing the performance of the trained model.

```text
evaluation/evaluate.py
```

The system can be extended with:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

## 🎯 Applications

Candy Speech can be useful for:

* Speech detection
* Audio classification
* Intelligent voice systems
* Environmental sound monitoring
* Human activity analysis
* Real-time audio analysis
* AI-based sound recognition

## 🔮 Future Enhancements

Future improvements may include:

* Real-time microphone streaming
* Deep learning-based CNN-RNN models
* Improved noise detection
* Larger and more diverse datasets
* Cloud deployment
* Mobile application integration
* Improved model accuracy
* Automatic audio event detection

## 👨‍💻 Author

**Sasikumar Pallipamu**

B.Tech – Information Technology

Aditya Engineering College / Aditya University

GitHub: [pallipamusasikumar-coder](https://github.com/pallipamusasikumar-coder)

## 📜 Project

**CandySpeech: An Intelligent Deep Learning Framework for Real-Time Speech Detection**

This project was developed as an academic/research project focusing on audio feature extraction, machine learning, and real-time sound classification.
