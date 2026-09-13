import os
import tempfile

import streamlit as st
import librosa
import librosa.display
import soundfile as sf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import load


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="CandySound AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "sound_model.joblib"
)

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset"
)


# =========================================================
# AUDIO / FEATURE SETTINGS
# =========================================================

SAMPLE_RATE = 22050
N_MFCC = 13
EXPECTED_FEATURES = 78


# =========================================================
# CATEGORIES
# =========================================================

DEFAULT_CATEGORIES = [
    "human",
    "animal",
    "noise"
]

CATEGORY_INFO = {

    "human": {
        "icon": "👤",
        "name": "Human",
        "description": (
            "Human voice or human-generated sound."
        )
    },

    "animal": {
        "icon": "🐕",
        "name": "Animal",
        "description": (
            "Animal-generated sound such as "
            "dog, cat or bird."
        )
    },

    "noise": {
        "icon": "🔊",
        "name": "Noise",
        "description": (
            "Environmental, machine or "
            "background noise."
        )
    }
}


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 45px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 19px;
        color: #777777;
        margin-bottom: 20px;
    }

    .result-box {
        padding: 28px;
        border-radius: 18px;
        text-align: center;
        font-size: 30px;
        font-weight: bold;
        margin-top: 15px;
        margin-bottom: 20px;
    }

    .human-box {
        background-color: #d4edda;
        color: #155724;
    }

    .animal-box {
        background-color: #fff3cd;
        color: #856404;
    }

    .noise-box {
        background-color: #f8d7da;
        color: #721c24;
    }

    .info-card {
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

model = None
categories = DEFAULT_CATEGORIES.copy()
model_accuracy = 0
model_status = False
model_error = None


try:

    model_data = load(MODEL_PATH)

    if isinstance(model_data, dict):

        model = model_data.get(
            "model"
        )

        categories = model_data.get(
            "categories",
            DEFAULT_CATEGORIES
        )

        model_accuracy = model_data.get(
            "accuracy",
            0
        )

    else:

        model = model_data

    model_status = model is not None


except Exception as e:

    model_status = False
    model_error = str(e)


# =========================================================
# AUDIO LOADER
# =========================================================

def load_audio(file_path):

    audio, sr = sf.read(
        file_path,
        always_2d=False
    )

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    # Stereo → Mono
    if audio.ndim > 1:

        audio = np.mean(
            audio,
            axis=1
        )

    # Resample
    if sr != SAMPLE_RATE:

        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=SAMPLE_RATE
        )

        sr = SAMPLE_RATE

    if len(audio) == 0:

        raise ValueError(
            "Audio file is empty."
        )

    return audio, sr


# =========================================================
# 78 FEATURE EXTRACTION
# =========================================================

def extract_features(file_path):

    audio, sr = load_audio(
        file_path
    )

    # -----------------------------------------
    # MFCC
    # 13
    # -----------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    # -----------------------------------------
    # Delta
    # 13
    # -----------------------------------------

    delta = librosa.feature.delta(
        mfcc
    )

    # -----------------------------------------
    # Delta-Delta
    # 13
    # -----------------------------------------

    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    # -----------------------------------------
    # 78 FEATURES
    #
    # MFCC mean       13
    # MFCC std        13
    # Delta mean      13
    # Delta std       13
    # Delta2 mean     13
    # Delta2 std      13
    #
    # TOTAL = 78
    # -----------------------------------------

    features = np.concatenate([

        np.mean(
            mfcc,
            axis=1
        ),

        np.std(
            mfcc,
            axis=1
        ),

        np.mean(
            delta,
            axis=1
        ),

        np.std(
            delta,
            axis=1
        ),

        np.mean(
            delta2,
            axis=1
        ),

        np.std(
            delta2,
            axis=1
        )

    ])

    features = np.asarray(
        features,
        dtype=np.float64
    ).flatten()

    if len(features) != EXPECTED_FEATURES:

        raise ValueError(
            f"Expected {EXPECTED_FEATURES} features "
            f"but generated {len(features)}."
        )

    return features


# =========================================================
# DATASET COUNTS
# =========================================================

def get_dataset_counts():

    counts = {}

    for category in DEFAULT_CATEGORIES:

        folder = os.path.join(
            DATASET_PATH,
            category
        )

        if os.path.exists(folder):

            count = len([
                f
                for f in os.listdir(folder)
                if f.lower().endswith(".wav")
            ])

        else:

            count = 0

        counts[category] = count

    return counts


dataset_counts = get_dataset_counts()

total_files = sum(
    dataset_counts.values()
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="main-title">
        🎙️ CandySound AI
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Intelligent Human, Animal & Noise Detection System
    </div>
    """,
    unsafe_allow_html=True
)

st.info(
    "Upload a WAV audio file and CandySound AI "
    "will analyze the audio using MFCC, Delta and "
    "Delta-Delta features."
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "⚙️ System Information"
)

if model_status:

    st.sidebar.success(
        "✅ Model Loaded"
    )

else:

    st.sidebar.error(
        "❌ Model Not Loaded"
    )

    if model_error:

        st.sidebar.caption(
            model_error
        )


st.sidebar.divider()

st.sidebar.write(
    "**Model:** Random Forest"
)

st.sidebar.write(
    "**Classes:** Human / Animal / Noise"
)

st.sidebar.write(
    "**Audio:** WAV"
)

st.sidebar.write(
    "**Sample Rate:** 22050 Hz"
)

st.sidebar.write(
    "**MFCC:** 13"
)

st.sidebar.write(
    "**Feature Type:** MFCC + Delta + Delta-Delta"
)

st.sidebar.write(
    "**Total Features:** 78"
)


if model is not None:

    model_features = getattr(
        model,
        "n_features_in_",
        "Unknown"
    )

    st.sidebar.write(
        f"**Model Features:** {model_features}"
    )


if model_accuracy:

    st.sidebar.write(
        f"**Test Accuracy:** "
        f"{model_accuracy * 100:.2f}%"
    )


st.sidebar.divider()

st.sidebar.write(
    "🎙️ CandySound AI v2.0"
)


# =========================================================
# DATASET STATISTICS
# =========================================================

st.subheader(
    "📊 Dataset Statistics"
)

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "🎵 Total Audio",
        total_files
    )


with col2:

    st.metric(
        "👤 Human",
        dataset_counts["human"]
    )


with col3:

    st.metric(
        "🐕 Animal",
        dataset_counts["animal"]
    )


with col4:

    st.metric(
        "🔊 Noise",
        dataset_counts["noise"]
    )


# =========================================================
# DATASET OVERVIEW
# =========================================================

st.divider()

st.subheader(
    "📈 Dataset Overview"
)

dataset_df = pd.DataFrame({

    "Category": [
        "Human",
        "Animal",
        "Noise"
    ],

    "Files": [
        dataset_counts["human"],
        dataset_counts["animal"],
        dataset_counts["noise"]
    ]
})


graph1, graph2 = st.columns(2)


# =========================================================
# BAR CHART
# =========================================================

with graph1:

    st.write(
        "### 📊 Files by Category"
    )

    st.bar_chart(
        dataset_df.set_index(
            "Category"
        )
    )


# =========================================================
# PIE CHART
# =========================================================

with graph2:

    st.write(
        "### 🥧 Dataset Distribution"
    )

    if total_files > 0:

        fig, ax = plt.subplots(
            figsize=(6, 5)
        )

        ax.pie(
            dataset_df["Files"],
            labels=dataset_df["Category"],
            autopct="%1.1f%%",
            startangle=90
        )

        ax.set_title(
            "Human vs Animal vs Noise"
        )

        st.pyplot(fig)

        plt.close(fig)

    else:

        st.warning(
            "No audio files found."
        )


# =========================================================
# DATASET TABLE
# =========================================================

st.subheader(
    "📋 Dataset Summary"
)

if total_files > 0:

    summary_df = dataset_df.copy()

    summary_df["Percentage"] = (
        summary_df["Files"]
        / total_files
        * 100
    )

    display_df = summary_df.copy()

    display_df["Percentage"] = (
        display_df["Percentage"]
        .map(
            lambda x:
            f"{x:.2f}%"
        )
    )

    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True
    )


# =========================================================
# MODEL STATUS
# =========================================================

st.divider()

st.subheader(
    "🤖 Model Status"
)

m1, m2, m3, m4 = st.columns(4)


with m1:

    if model_status:

        st.success(
            "✅ Model Ready"
        )

    else:

        st.error(
            "❌ Model Error"
        )


with m2:

    if model is not None:

        model_features = getattr(
            model,
            "n_features_in_",
            "Unknown"
        )

        st.metric(
            "Model Features",
            model_features
        )

    else:

        st.metric(
            "Model Features",
            "N/A"
        )


with m3:

    st.metric(
        "Dashboard Features",
        EXPECTED_FEATURES
    )


with m4:

    if model_accuracy:

        st.metric(
            "Test Accuracy",
            f"{model_accuracy * 100:.2f}%"
        )

    else:

        st.metric(
            "Test Accuracy",
            "N/A"
        )


# =========================================================
# AUDIO CLASSIFICATION
# =========================================================

st.divider()

st.subheader(
    "🎵 Audio Classification"
)


uploaded_file = st.file_uploader(
    "Upload a WAV audio file",
    type=["wav"],
    help="Upload a WAV file for analysis."
)


if uploaded_file is not None:

    # =====================================================
    # TEMP FILE
    # =====================================================

    suffix = ".wav"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp:

        temp.write(
            uploaded_file.getbuffer()
        )

        temp_file = temp.name


    try:

        # =================================================
        # AUDIO PLAYER
        # =================================================

        st.write(
            "### 🔊 Audio Player"
        )

        st.audio(
            uploaded_file,
            format="audio/wav"
        )


        # =================================================
        # LOAD AUDIO
        # =================================================

        audio, sr = load_audio(
            temp_file
        )

        duration = librosa.get_duration(
            y=audio,
            sr=sr
        )


        # =================================================
        # RMS
        # =================================================

        rms_values = librosa.feature.rms(
            y=audio
        )

        rms = float(
            np.mean(rms_values)
        )


        # =================================================
        # ZCR
        # =================================================

        zcr_values = librosa.feature.zero_crossing_rate(
            audio
        )

        zcr = float(
            np.mean(zcr_values)
        )


        # =================================================
        # AUDIO INFORMATION
        # =================================================

        st.subheader(
            "📋 Audio Information"
        )

        a1, a2, a3, a4 = st.columns(4)


        with a1:

            st.metric(
                "📄 File",
                uploaded_file.name
            )


        with a2:

            st.metric(
                "⏱️ Duration",
                f"{duration:.2f}s"
            )


        with a3:

            st.metric(
                "📡 Sample Rate",
                f"{sr} Hz"
            )


        with a4:

            st.metric(
                "📦 Size",
                f"{uploaded_file.size / 1024:.2f} KB"
            )


        # =================================================
        # ACOUSTIC FEATURES
        # =================================================

        st.subheader(
            "🎚️ Acoustic Features"
        )

        b1, b2, b3 = st.columns(3)


        with b1:

            st.metric(
                "⚡ RMS Energy",
                f"{rms:.5f}"
            )


        with b2:

            st.metric(
                "🔄 Zero Crossing Rate",
                f"{zcr:.5f}"
            )


        with b3:

            st.metric(
                "🎵 MFCC Count",
                N_MFCC
            )


        # =================================================
        # WAVEFORM
        # =================================================

        st.divider()

        st.subheader(
            "🌊 Audio Waveform"
        )

        fig_wave, ax_wave = plt.subplots(
            figsize=(12, 4)
        )

        librosa.display.waveshow(
            audio,
            sr=sr,
            ax=ax_wave
        )

        ax_wave.set_title(
            "Audio Waveform"
        )

        ax_wave.set_xlabel(
            "Time (seconds)"
        )

        ax_wave.set_ylabel(
            "Amplitude"
        )

        st.pyplot(fig_wave)

        plt.close(fig_wave)


        # =================================================
        # SPECTROGRAM
        # =================================================

        st.subheader(
            "🌈 Spectrogram"
        )

        stft = librosa.stft(
            audio
        )

        magnitude = np.abs(
            stft
        )

        spectrogram = librosa.amplitude_to_db(
            magnitude,
            ref=np.max
        )

        fig_spec, ax_spec = plt.subplots(
            figsize=(12, 5)
        )

        image = librosa.display.specshow(
            spectrogram,
            sr=sr,
            x_axis="time",
            y_axis="hz",
            ax=ax_spec
        )

        ax_spec.set_title(
            "Audio Spectrogram"
        )

        fig_spec.colorbar(
            image,
            ax=ax_spec,
            format="%+2.0f dB"
        )

        st.pyplot(fig_spec)

        plt.close(fig_spec)


        # =================================================
        # ANALYZE BUTTON
        # =================================================

        st.divider()

        analyze = st.button(
            "🔍 Analyze Sound",
            type="primary",
            use_container_width=True
        )


        if analyze:

            if model is None:

                st.error(
                    "Model is not available. "
                    "Please train the model first."
                )

            else:

                # =========================================
                # EXTRACT FEATURES
                # =========================================

                features = extract_features(
                    temp_file
                )

                feature_count = len(
                    features
                )

                expected_features = getattr(
                    model,
                    "n_features_in_",
                    EXPECTED_FEATURES
                )


                # =========================================
                # FEATURE INFORMATION
                # =========================================

                st.subheader(
                    "🧠 Feature Extraction"
                )

                fc1, fc2, fc3 = st.columns(3)


                with fc1:

                    st.metric(
                        "MFCC",
                        "13"
                    )


                with fc2:

                    st.metric(
                        "Generated Features",
                        feature_count
                    )


                with fc3:

                    st.metric(
                        "Model Expected",
                        expected_features
                    )


                # =========================================
                # FEATURE CHECK
                # =========================================

                if feature_count != expected_features:

                    st.error(
                        f"""
                        ❌ Feature mismatch.

                        Model expects: {expected_features}

                        Dashboard generates: {feature_count}

                        Please retrain the model using
                        the same 78-feature extraction method.
                        """
                    )

                    st.stop()


                st.success(
                    f"✅ Feature extraction matched: "
                    f"{feature_count} features"
                )


                # =========================================
                # PREDICTION
                # =========================================

                prediction = model.predict(
                    [features]
                )

                predicted_value = prediction[0]


                # =========================================
                # CLASS INDEX
                # =========================================

                class_index = None

                try:

                    class_index = int(
                        predicted_value
                    )

                except:

                    pass


                # =========================================
                # CATEGORY
                # =========================================

                if class_index is not None:

                    if (
                        0 <= class_index
                        < len(categories)
                    ):

                        predicted_category = str(
                            categories[class_index]
                        ).lower()

                    else:

                        predicted_category = "noise"

                else:

                    predicted_category = str(
                        predicted_value
                    ).lower()


                # =========================================
                # NORMALIZE CATEGORY
                # =========================================

                if "human" in predicted_category:

                    predicted_category = "human"

                elif "animal" in predicted_category:

                    predicted_category = "animal"

                elif "noise" in predicted_category:

                    predicted_category = "noise"


                # =========================================
                # CONFIDENCE
                # =========================================

                confidence = None
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

                    confidence = (
                        float(
                            np.max(
                                probabilities
                            )
                        )
                        * 100
                    )


                # =========================================
                # CATEGORY INFO
                # =========================================

                info = CATEGORY_INFO.get(
                    predicted_category,
                    {
                        "icon": "🎵",
                        "name":
                            predicted_category.title(),
                        "description":
                            "Detected audio category."
                    }
                )

                icon = info["icon"]

                name = info["name"]

                description = info[
                    "description"
                ]


                # =========================================
                # RESULT STYLE
                # =========================================

                if predicted_category == "human":

                    box_class = "human-box"

                elif predicted_category == "animal":

                    box_class = "animal-box"

                else:

                    box_class = "noise-box"


                # =========================================
                # FINAL RESULT
                # =========================================

                st.markdown(
                    f"""
                    <div class="result-box {box_class}">
                        {icon} {name.upper()}
                        SOUND DETECTED
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.write(
                    f"**Description:** {description}"
                )


                # =========================================
                # CONFIDENCE
                # =========================================

                if confidence is not None:

                    st.subheader(
                        "🎯 Prediction Confidence"
                    )

                    conf1, conf2 = st.columns(2)


                    with conf1:

                        st.metric(
                            "Confidence",
                            f"{confidence:.2f}%"
                        )


                    with conf2:

                        st.progress(
                            min(
                                confidence / 100,
                                1.0
                            )
                        )


                # =========================================
                # CLASS PROBABILITIES
                # =========================================

                if probabilities is not None:

                    st.divider()

                    st.subheader(
                        "📊 Class Probabilities"
                    )

                    probability_rows = []


                    for i, probability in enumerate(
                        probabilities
                    ):

                        if i < len(categories):

                            category_name = str(
                                categories[i]
                            ).title()

                        else:

                            category_name = (
                                f"Class {i}"
                            )


                        probability_rows.append({

                            "Category":
                                category_name,

                            "Probability":
                                probability * 100

                        })


                    probability_df = pd.DataFrame(
                        probability_rows
                    )


                    st.bar_chart(
                        probability_df.set_index(
                            "Category"
                        )
                    )


                    probability_display = (
                        probability_df.copy()
                    )

                    probability_display[
                        "Probability"
                    ] = (
                        probability_display[
                            "Probability"
                        ]
                        .map(
                            lambda x:
                            f"{x:.2f}%"
                        )
                    )


                    st.dataframe(
                        probability_display,
                        hide_index=True,
                        use_container_width=True
                    )


                # =========================================
                # MFCC ANALYSIS
                # =========================================

                st.divider()

                st.subheader(
                    "🧠 MFCC Analysis"
                )

                mfcc = librosa.feature.mfcc(
                    y=audio,
                    sr=sr,
                    n_mfcc=N_MFCC
                )

                mfcc_mean = np.mean(
                    mfcc,
                    axis=1
                )

                mfcc_names = [

                    f"MFCC {i + 1}"

                    for i in range(
                        N_MFCC
                    )

                ]

                mfcc_df = pd.DataFrame({

                    "MFCC":
                        mfcc_names,

                    "Value":
                        mfcc_mean

                })


                st.bar_chart(
                    mfcc_df.set_index(
                        "MFCC"
                    )
                )


                # =========================================
                # FEATURE GROUP SUMMARY
                # =========================================

                st.subheader(
                    "📊 Feature Group Summary"
                )

                feature_groups = pd.DataFrame({

                    "Feature Group": [
                        "MFCC Mean",
                        "MFCC Std",
                        "Delta Mean",
                        "Delta Std",
                        "Delta-Delta Mean",
                        "Delta-Delta Std"
                    ],

                    "Number of Features": [
                        13,
                        13,
                        13,
                        13,
                        13,
                        13
                    ]

                })

                st.dataframe(
                    feature_groups,
                    hide_index=True,
                    use_container_width=True
                )


                # =========================================
                # ALL 78 FEATURES
                # =========================================

                st.subheader(
                    "📋 All 78 Extracted Features"
                )

                feature_names = []

                for i in range(13):

                    feature_names.append(
                        f"MFCC Mean {i + 1}"
                    )

                for i in range(13):

                    feature_names.append(
                        f"MFCC Std {i + 1}"
                    )

                for i in range(13):

                    feature_names.append(
                        f"Delta Mean {i + 1}"
                    )

                for i in range(13):

                    feature_names.append(
                        f"Delta Std {i + 1}"
                    )

                for i in range(13):

                    feature_names.append(
                        f"Delta-Delta Mean {i + 1}"
                    )

                for i in range(13):

                    feature_names.append(
                        f"Delta-Delta Std {i + 1}"
                    )


                feature_df = pd.DataFrame({

                    "Feature":
                        feature_names,

                    "Value":
                        features

                })


                st.dataframe(
                    feature_df,
                    hide_index=True,
                    use_container_width=True
                )


                # =========================================
                # PREDICTION DETAILS
                # =========================================

                st.subheader(
                    "📄 Prediction Details"
                )

                details_df = pd.DataFrame({

                    "Property": [

                        "Audio File",

                        "Detected Category",

                        "Confidence",

                        "Duration",

                        "Sample Rate",

                        "File Size",

                        "RMS Energy",

                        "Zero Crossing Rate",

                        "MFCC Features",

                        "Total Features",

                        "Model Features",

                        "Model"

                    ],

                    "Value": [

                        uploaded_file.name,

                        name,

                        (
                            f"{confidence:.2f}%"
                            if confidence is not None
                            else "N/A"
                        ),

                        f"{duration:.2f} sec",

                        f"{sr} Hz",

                        f"{uploaded_file.size / 1024:.2f} KB",

                        f"{rms:.5f}",

                        f"{zcr:.5f}",

                        "13",

                        str(
                            len(features)
                        ),

                        str(
                            expected_features
                        ),

                        "Random Forest"

                    ]

                })


                st.dataframe(
                    details_df,
                    hide_index=True,
                    use_container_width=True
                )


                # =========================================
                # DOWNLOAD REPORT
                # =========================================

                report = f"""
CANDYSOUND AI
========================================

AUDIO ANALYSIS REPORT
========================================

File Name:
{uploaded_file.name}

Prediction:
{name}

Confidence:
{
    f"{confidence:.2f}%"
    if confidence is not None
    else "N/A"
}

Description:
{description}

Audio Duration:
{duration:.2f} seconds

Sample Rate:
{sr} Hz

File Size:
{uploaded_file.size / 1024:.2f} KB

RMS Energy:
{rms:.5f}

Zero Crossing Rate:
{zcr:.5f}

MFCC Features:
13

Delta Features:
13

Delta-Delta Features:
13

Statistics per Feature:
Mean + Standard Deviation

Total Features:
{len(features)}

Model Expected Features:
{expected_features}

Model:
Random Forest

Prediction System:
Human / Animal / Noise

========================================
Generated by CandySound AI
========================================
"""


                st.download_button(

                    "📥 Download Analysis Report",

                    data=report,

                    file_name=
                        "candysound_report.txt",

                    mime="text/plain",

                    use_container_width=True

                )


    except Exception as e:

        st.error(
            f"❌ Audio processing error: {e}"
        )


    finally:

        # Remove temporary file
        if os.path.exists(temp_file):

            try:

                os.remove(temp_file)

            except:

                pass


# =========================================================
# AI PROCESSING PIPELINE
# =========================================================

st.divider()

st.subheader(
    "🔄 AI Processing Pipeline"
)

w1, w2, w3, w4 = st.columns(4)


with w1:

    st.write(
        "### 🎵 1. Input"
    )

    st.write(
        "WAV audio is uploaded."
    )


with w2:

    st.write(
        "### 🧠 2. Features"
    )

    st.write(
        "13 MFCC + Delta + "
        "Delta-Delta features."
    )


with w3:

    st.write(
        "### 🤖 3. Classification"
    )

    st.write(
        "Random Forest analyzes "
        "78 audio features."
    )


with w4:

    st.write(
        "### 🎯 4. Result"
    )

    st.write(
        "Human, Animal or Noise."
    )


# =========================================================
# SOUND CATEGORIES
# =========================================================

st.divider()

st.subheader(
    "🔎 Sound Categories"
)

cat1, cat2, cat3 = st.columns(3)


with cat1:

    st.write(
        "### 👤 Human"
    )

    st.write(
        "Speech, talking, singing, "
        "laughing and other human sounds."
    )


with cat2:

    st.write(
        "### 🐕 Animal"
    )

    st.write(
        "Dog, cat, bird and other "
        "animal sounds."
    )


with cat3:

    st.write(
        "### 🔊 Noise"
    )

    st.write(
        "Traffic, engines, machines, "
        "fans and environmental sounds."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🎙️ CandySound AI | "
    "Human • Animal • Noise Classification | "
    "MFCC + Delta + Delta-Delta | 78 Features"
)