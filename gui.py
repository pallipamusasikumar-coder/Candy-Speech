import streamlit as st
import librosa
import numpy as np
import joblib
import os
import pandas as pd


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="CandySpeech Dashboard",
    page_icon="🎙️",
    layout="wide"
)


# ==========================================
# LOAD MODEL
# ==========================================

MODEL_PATH = "speech_model.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
    model_status = True
except Exception as e:
    model = None
    model_status = False


# ==========================================
# FEATURE EXTRACTION
# ==========================================

def extract_features(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=22050
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13
    )

    features = np.mean(
        mfcc.T,
        axis=0
    )

    return features


# ==========================================
# DATASET INFORMATION
# ==========================================

speech_folder = os.path.join(
    "dataset",
    "speech"
)

non_speech_folder = os.path.join(
    "dataset",
    "non_speech"
)


def count_audio_files(folder):

    if not os.path.exists(folder):
        return 0

    return len([
        f for f in os.listdir(folder)
        if f.lower().endswith(".wav")
    ])


speech_count = count_audio_files(
    speech_folder
)

non_speech_count = count_audio_files(
    non_speech_folder
)

total_count = speech_count + non_speech_count


# ==========================================
# HEADER
# ==========================================

st.title("🎙️ CandySpeech")

st.subheader(
    "Intelligent Speech vs Non-Speech Detection Dashboard"
)

st.write(
    "Machine Learning based audio classification using MFCC features."
)


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🎛️ Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "Audio Prediction",
        "Dataset",
        "About"
    ]
)


# ==========================================
# DASHBOARD
# ==========================================

if page == "Dashboard":

    st.header("📊 System Dashboard")

    # Model status

    if model_status:

        st.success(
            "🟢 Speech Detection Model Loaded"
        )

    else:

        st.error(
            "🔴 Speech Detection Model Not Found"
        )


    # Metrics

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Audio",
            total_count
        )

    with col2:

        st.metric(
            "Speech",
            speech_count
        )

    with col3:

        st.metric(
            "Non-Speech",
            non_speech_count
        )

    with col4:

        st.metric(
            "MFCC Features",
            13
        )


    st.divider()


    # Dataset chart

    st.subheader(
        "📈 Dataset Distribution"
    )

    chart_data = pd.DataFrame(
        {
            "Category": [
                "Speech",
                "Non-Speech"
            ],
            "Files": [
                speech_count,
                non_speech_count
            ]
        }
    )

    st.bar_chart(
        chart_data.set_index("Category")
    )


# ==========================================
# AUDIO PREDICTION
# ==========================================

elif page == "Audio Prediction":

    st.header(
        "🔊 Audio Prediction"
    )

    st.write(
        "Upload a WAV file and the trained model will classify it."
    )


    uploaded_file = st.file_uploader(
        "Choose a WAV audio file",
        type=["wav"]
    )


    if uploaded_file is not None:

        st.audio(
            uploaded_file,
            format="audio/wav"
        )

        st.write(
            "**Audio:**",
            uploaded_file.name
        )


        if st.button(
            "🔍 Predict Audio"
        ):

            if not model_status:

                st.error(
                    "Model is not available."
                )

            else:

                try:

                    # Save uploaded file

                    temp_file = "temp_audio.wav"

                    with open(
                        temp_file,
                        "wb"
                    ) as f:

                        f.write(
                            uploaded_file.getbuffer()
                        )


                    # Extract MFCC

                    features = extract_features(
                        temp_file
                    )


                    # Prediction

                    prediction = model.predict(
                        [features]
                    )[0]


                    # Probability

                    probabilities = None

                    if hasattr(
                        model,
                        "predict_proba"
                    ):

                        probabilities = (
                            model.predict_proba(
                                [features]
                            )[0]
                        )


                    # Remove temporary file

                    if os.path.exists(
                        temp_file
                    ):

                        os.remove(
                            temp_file
                        )


                    st.divider()


                    # RESULT

                    if prediction == 1:

                        st.success(
                            "🗣️ SPEECH DETECTED"
                        )

                        st.subheader(
                            "Prediction: Speech"
                        )

                    else:

                        st.warning(
                            "🔇 NON-SPEECH DETECTED"
                        )

                        st.subheader(
                            "Prediction: Non-Speech"
                        )


                    # CONFIDENCE

                    if probabilities is not None:

                        st.subheader(
                            "📊 Prediction Confidence"
                        )


                        speech_probability = 0
                        non_speech_probability = 0


                        for i, cls in enumerate(
                            model.classes_
                        ):

                            if cls == 1:

                                speech_probability = (
                                    probabilities[i]
                                    * 100
                                )

                            elif cls == 0:

                                non_speech_probability = (
                                    probabilities[i]
                                    * 100
                                )


                        c1, c2 = st.columns(2)


                        with c1:

                            st.metric(
                                "Speech",
                                f"{speech_probability:.2f}%"
                            )


                        with c2:

                            st.metric(
                                "Non-Speech",
                                f"{non_speech_probability:.2f}%"
                            )


                        st.progress(
                            int(
                                speech_probability
                            )
                        )


                except Exception as e:

                    st.error(
                        f"Prediction error: {e}"
                    )


# ==========================================
# DATASET
# ==========================================

elif page == "Dataset":

    st.header(
        "📁 Dataset"
    )


    st.write(
        "Current dataset statistics"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.info(
            f"🗣️ Speech Files: {speech_count}"
        )


    with col2:

        st.warning(
            f"🔇 Non-Speech Files: {non_speech_count}"
        )


    st.subheader(
        "Dataset Structure"
    )


    st.code(
        """
dataset/
│
├── speech/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
│
└── non_speech/
    ├── audio1.wav
    ├── audio2.wav
    └── ...
        """
    )


# ==========================================
# ABOUT
# ==========================================

elif page == "About":

    st.header(
        "ℹ️ About CandySpeech"
    )


    st.write(
        """
        CandySpeech is an intelligent audio classification
        system designed to distinguish between speech and
        non-speech audio.
        """
    )


    st.subheader(
        "Technology Stack"
    )


    st.markdown(
        """
        - 🐍 Python
        - 🎵 Librosa
        - 🧠 Scikit-learn
        - 📊 NumPy
        - 💾 Joblib
        - 🌐 Streamlit
        - 🎙️ MFCC Audio Features
        """
    )


    st.subheader(
        "Machine Learning Pipeline"
    )


    st.code(
        """
Audio
   ↓
MFCC Feature Extraction
   ↓
Machine Learning Model
   ↓
Prediction
   ↓
Speech / Non-Speech
        """
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "CandySpeech | Intelligent Speech Detection System"
)