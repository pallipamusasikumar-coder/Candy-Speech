import os
import streamlit as st
import librosa
import librosa.display
import soundfile as sf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import load


# =========================================================
# PAGE CONFIGURATION
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
# CLASS INFORMATION
# =========================================================

CATEGORIES = [
    "human",
    "animal",
    "noise"
]


CATEGORY_INFO = {

    "human": {
        "icon": "👤",
        "name": "Human",
        "description":
            "Human voice or human-generated sound."
    },

    "animal": {
        "icon": "🐕",
        "name": "Animal",
        "description":
            "Animal-generated sound such as dog, cat or bird."
    },

    "noise": {
        "icon": "🔊",
        "name": "Noise",
        "description":
            "Environmental, machine or background noise."
    }
}


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 46px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 19px;
        color: #777777;
        margin-bottom: 20px;
    }

    .result-box {
        padding: 30px;
        border-radius: 18px;
        text-align: center;
        font-size: 30px;
        font-weight: bold;
        margin: 20px 0;
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
        padding: 20px;
        border-radius: 15px;
        background-color: #f5f5f5;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

model = None
categories = CATEGORIES.copy()
model_accuracy = 0.0
model_loaded = False
model_error = None


try:

    model_data = load(
        MODEL_PATH
    )

    if isinstance(
        model_data,
        dict
    ):

        model = model_data.get(
            "model"
        )

        categories = model_data.get(
            "categories",
            CATEGORIES
        )

        model_accuracy = model_data.get(
            "accuracy",
            0.0
        )

    else:

        model = model_data

    if model is not None:

        model_loaded = True


except Exception as e:

    model_error = str(e)
    model_loaded = False


# =========================================================
# AUDIO LOADING FUNCTION
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

    # Resampling

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

    # -----------------------------------------------------
    # MFCC
    # 13 features
    # -----------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    # -----------------------------------------------------
    # DELTA
    # 13 features
    # -----------------------------------------------------

    delta = librosa.feature.delta(
        mfcc
    )

    # -----------------------------------------------------
    # DELTA DELTA
    # 13 features
    # -----------------------------------------------------

    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    # -----------------------------------------------------
    # 78 FEATURES
    # -----------------------------------------------------

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
            f"but generated {len(features)}"
        )

    return features


# =========================================================
# DATASET COUNTS
# =========================================================

def get_dataset_counts():

    counts = {}

    for category in CATEGORIES:

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
    "will analyze and classify the sound."
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "⚙️ System Information"
)


if model_loaded:

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


if model_accuracy > 0:

    st.sidebar.write(
        f"**Test Accuracy:** "
        f"{model_accuracy * 100:.2f}%"
    )


st.sidebar.divider()

st.sidebar.caption(
    "🎙️ CandySound AI v2.0"
)


# =========================================================
# DATASET STATISTICS
# =========================================================

st.subheader(
    "📊 Dataset Statistics"
)


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "🎵 Total Audio",
        total_files
    )


with c2:

    st.metric(
        "👤 Human",
        dataset_counts["human"]
    )


with c3:

    st.metric(
        "🐕 Animal",
        dataset_counts["animal"]
    )


with c4:

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


chart1, chart2 = st.columns(2)


# =========================================================
# BAR CHART
# =========================================================

with chart1:

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

with chart2:

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

        st.pyplot(
            fig
        )

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


m1, m2, m3 = st.columns(3)


with m1:

    if model_loaded:

        st.success(
            "✅ Model Ready"
        )

    else:

        st.error(
            "❌ Model Error"
        )


with m2:

    if model is not None:

        features = getattr(
            model,
            "n_features_in_",
            "N/A"
        )

        st.metric(
            "Model Features",
            features
        )

    else:

        st.metric(
            "Model Features",
            "N/A"
        )


with m3:

    if model_accuracy > 0:

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
# AUDIO UPLOAD
# =========================================================

st.divider()

st.subheader(
    "🎵 Audio Classification"
)


uploaded_file = st.file_uploader(
    "Upload a WAV audio file",
    type=["wav"],
    help="Upload a WAV file for classification."
)


if uploaded_file is not None:

    # -----------------------------------------------------
    # TEMP FILE
    # -----------------------------------------------------

    temp_file = os.path.join(
        PROJECT_ROOT,
        "_temp_audio.wav"
    )

    with open(
        temp_file,
        "wb"
    ) as f:

        f.write(
            uploaded_file.getbuffer()
        )


    # -----------------------------------------------------
    # AUDIO PLAYER
    # -----------------------------------------------------

    st.write(
        "### 🔊 Audio Player"
    )

    st.audio(
        uploaded_file,
        format="audio/wav"
    )


    try:

        # -------------------------------------------------
        # LOAD AUDIO
        # -------------------------------------------------

        audio, sr = load_audio(
            temp_file
        )


        # -------------------------------------------------
        # DURATION
        # -------------------------------------------------

        duration = librosa.get_duration(
            y=audio,
            sr=sr
        )


        # -------------------------------------------------
        # RMS ENERGY
        # -------------------------------------------------

        rms_values = librosa.feature.rms(
            y=audio
        )

        rms = float(
            np.mean(rms_values)
        )


        # -------------------------------------------------
        # ZERO CROSSING RATE
        # -------------------------------------------------

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


        b1, b2 = st.columns(2)


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


        st.pyplot(
            fig_wave,
            use_container_width=True
        )

        plt.close(
            fig_wave
        )


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


        st.pyplot(
            fig_spec,
            use_container_width=True
        )

        plt.close(
            fig_spec
        )


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
                    "Run train_model.py first."
                )

            else:

                try:

                    # =====================================
                    # FEATURE EXTRACTION
                    # =====================================

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


                    # =====================================
                    # FEATURE STATUS
                    # =====================================

                    fc1, fc2 = st.columns(2)


                    with fc1:

                        st.metric(
                            "Generated Features",
                            feature_count
                        )


                    with fc2:

                        st.metric(
                            "Model Expected",
                            expected_features
                        )


                    # =====================================
                    # FEATURE VALIDATION
                    # =====================================

                    if feature_count != expected_features:

                        st.error(
                            f"""
                            ❌ Feature mismatch.

                            Model expects:
                            {expected_features}

                            Dashboard generated:
                            {feature_count}

                            Please retrain the model using
                            the same feature extraction code.
                            """
                        )

                        st.stop()


                    st.success(
                        "✅ Feature extraction matched: "
                        f"{feature_count} features"
                    )


                    # =====================================
                    # PREDICTION
                    # =====================================

                    prediction = model.predict(
                        [features]
                    )


                    predicted_index = int(
                        prediction[0]
                    )


                    # =====================================
                    # CLASS MAPPING
                    # =====================================

                    if (
                        0 <= predicted_index
                        < len(categories)
                    ):

                        predicted_category = str(
                            categories[
                                predicted_index
                            ]
                        ).lower()

                    else:

                        predicted_category = "noise"


                    # =====================================
                    # PROBABILITY
                    # =====================================

                    probabilities = None
                    confidence = None


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
                            ) * 100
                        )


                    # =====================================
                    # CATEGORY INFORMATION
                    # =====================================

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


                    # =====================================
                    # RESULT COLOR
                    # =====================================

                    if predicted_category == "human":

                        result_class = "human-box"

                    elif predicted_category == "animal":

                        result_class = "animal-box"

                    else:

                        result_class = "noise-box"


                    # =====================================
                    # RESULT
                    # =====================================

                    st.markdown(
                        f"""
                        <div class="result-box {result_class}">
                            {icon} {name.upper()}
                            SOUND DETECTED
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                    st.write(
                        f"**Description:** {description}"
                    )


                    # =====================================
                    # CONFIDENCE
                    # =====================================

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


                    # =====================================
                    # CLASS PROBABILITIES
                    # =====================================

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


                    # =====================================
                    # MFCC ANALYSIS
                    # =====================================

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


                    mfcc_df = pd.DataFrame({

                        "MFCC": [
                            f"MFCC {i + 1}"
                            for i in range(N_MFCC)
                        ],

                        "Value":
                            mfcc_mean

                    })


                    st.bar_chart(
                        mfcc_df.set_index(
                            "MFCC"
                        )
                    )


                    # =====================================
                    # ALL 78 FEATURES
                    # =====================================

                    st.subheader(
                        "📋 Extracted 78 Features"
                    )


                    feature_df = pd.DataFrame({

                        "Feature": [
                            f"Feature {i + 1}"
                            for i in range(
                                len(features)
                            )
                        ],

                        "Value":
                            features

                    })


                    st.dataframe(
                        feature_df,
                        hide_index=True,
                        use_container_width=True
                    )


                    # =====================================
                    # PREDICTION DETAILS
                    # =====================================

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
                            "Feature Count",
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

                            str(
                                len(features)
                            ),

                            "Random Forest"

                        ]

                    })


                    st.dataframe(
                        details_df,
                        hide_index=True,
                        use_container_width=True
                    )


                    # =====================================
                    # REPORT
                    # =====================================

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

Feature Count:
{len(features)}

Model Expected Features:
{expected_features}

Model:
Random Forest

========================================
Generated by CandySound AI
========================================
"""


                    st.download_button(
                        "📥 Download Analysis Report",
                        data=report,
                        file_name="candysound_report.txt",
                        mime="text/plain",
                        use_container_width=True
                    )


                except Exception as e:

                    st.error(
                        f"❌ Prediction error: {e}"
                    )


    except Exception as e:

        st.error(
            f"❌ Audio processing error: {e}"
        )


# =========================================================
# AI PROCESSING PIPELINE
# =========================================================

st.divider()

st.subheader(
    "🔄 AI Processing Pipeline"
)


p1, p2, p3, p4 = st.columns(4)


with p1:

    st.write(
        "### 🎵 1. Input"
    )

    st.write(
        "WAV audio is uploaded."
    )


with p2:

    st.write(
        "### 🧠 2. Features"
    )

    st.write(
        "MFCC + Delta + Delta-Delta "
        "features are extracted."
    )


with p3:

    st.write(
        "### 🤖 3. Classification"
    )

    st.write(
        "Random Forest analyzes "
        "78 acoustic features."
    )


with p4:

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
        "Speech, talking, singing, laughing "
        "and other human sounds."
    )


with cat2:

    st.write(
        "### 🐕 Animal"
    )

    st.write(
        "Dog, cat, bird and other animal sounds."
    )


with cat3:

    st.write(
        "### 🔊 Noise"
    )

    st.write(
        "Traffic, engines, machines, fans "
        "and environmental sounds."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🎙️ CandySound AI | "
    "Human • Animal • Noise Classification | "
    "78 Acoustic Features"
)