import os
import sys

import librosa
import numpy as np
from joblib import load


# =========================================================
# PROJECT PATH
# =========================================================

REALTIME_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    REALTIME_DIR
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "sound_model.joblib"
)


# =========================================================
# LOAD MODEL
# =========================================================

print()
print("=" * 55)
print("          CANDYSOUND AI PREDICTION")
print("=" * 55)
print()

if not os.path.exists(MODEL_PATH):

    print("❌ Model not found:")
    print(MODEL_PATH)
    sys.exit()


model = load(MODEL_PATH)

print("✅ Model loaded successfully")
print("📁 Model:", MODEL_PATH)


# =========================================================
# CLASSES
# =========================================================

CLASSES = {
    0: "Human",
    1: "Animal",
    2: "Noise"
}


# =========================================================
# FEATURE EXTRACTION
# IMPORTANT:
# This MUST match training/train.py
# =========================================================

def extract_features(file_path):

    try:

        audio, sr = librosa.load(
            file_path,
            sr=22050,
            mono=True
        )

        if len(audio) == 0:

            raise ValueError(
                "Audio file is empty."
            )

        # -------------------------------------------------
        # MFCC
        # -------------------------------------------------

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=13
        )

        # -------------------------------------------------
        # Delta
        # -------------------------------------------------

        delta = librosa.feature.delta(
            mfcc
        )

        # -------------------------------------------------
        # Delta Delta
        # -------------------------------------------------

        delta2 = librosa.feature.delta(
            mfcc,
            order=2
        )

        # -------------------------------------------------
        # SAME 78 FEATURES AS TRAINING
        #
        # MFCC mean       = 13
        # MFCC std        = 13
        # Delta mean      = 13
        # Delta std       = 13
        # Delta2 mean     = 13
        # Delta2 std      = 13
        #
        # TOTAL           = 78
        # -------------------------------------------------

        features = np.concatenate(
            [

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

            ]
        )

        return features

    except Exception as e:

        print()
        print(
            "❌ Feature extraction error:"
        )

        print(e)

        return None


# =========================================================
# AUDIO FILE INPUT
# =========================================================

print()
audio_file = input(
    "🎵 Enter WAV audio file path: "
).strip()


# Remove quotes if user pastes path like:
# "C:\folder\audio.wav"

audio_file = audio_file.strip('"')
audio_file = audio_file.strip("'")


# =========================================================
# CHECK FILE
# =========================================================

if not os.path.exists(audio_file):

    print()
    print("❌ Audio file not found:")
    print(audio_file)
    sys.exit()


if not audio_file.lower().endswith(".wav"):

    print()
    print("❌ Please provide a WAV file.")
    sys.exit()


print()
print("📂 Audio:", audio_file)


# =========================================================
# EXTRACT FEATURES
# =========================================================

print()
print("🧠 Extracting audio features...")

features = extract_features(
    audio_file
)


if features is None:

    sys.exit()


print(
    f"✅ Features extracted: {len(features)}"
)


# =========================================================
# CHECK MODEL FEATURE COUNT
# =========================================================

expected_features = getattr(
    model,
    "n_features_in_",
    None
)


if expected_features is not None:

    print(
        f"🤖 Model expects: "
        f"{expected_features} features"
    )

    print(
        f"📊 Generated: "
        f"{len(features)} features"
    )


    if len(features) != expected_features:

        print()
        print("❌ FEATURE MISMATCH")
        print(
            f"Model expects {expected_features}, "
            f"but prediction generated "
            f"{len(features)}."
        )

        print()
        print(
            "Make sure realtime/predict.py uses "
            "the same feature extraction as "
            "training/train.py."
        )

        sys.exit()


# =========================================================
# PREDICTION
# =========================================================

print()
print("🤖 Analyzing sound...")


prediction = model.predict(
    [features]
)


predicted_index = int(
    prediction[0]
)


# =========================================================
# RESULT
# =========================================================

predicted_class = CLASSES.get(
    predicted_index,
    "Unknown"
)


print()
print("=" * 55)
print("                RESULT")
print("=" * 55)
print()


if predicted_class == "Human":

    print("👤 Prediction: HUMAN SOUND")

elif predicted_class == "Animal":

    print("🐕 Prediction: ANIMAL SOUND")

elif predicted_class == "Noise":

    print("🔊 Prediction: NOISE")

else:

    print(
        f"🎵 Prediction: {predicted_class}"
    )


# =========================================================
# CONFIDENCE
# =========================================================

if hasattr(
    model,
    "predict_proba"
):

    probabilities = model.predict_proba(
        [features]
    )[0]

    confidence = (
        float(
            np.max(
                probabilities
            )
        ) * 100
    )

    print()
    print(
        f"🎯 Confidence: "
        f"{confidence:.2f}%"
    )

    print()
    print("📊 Class Probabilities:")

    for index, probability in enumerate(
        probabilities
    ):

        class_name = CLASSES.get(
            index,
            f"Class {index}"
        )

        print(
            f"   {class_name:<10}: "
            f"{probability * 100:.2f}%"
        )


# =========================================================
# END
# =========================================================

print()
print("=" * 55)
print("           ANALYSIS COMPLETED")
print("=" * 55)
print()