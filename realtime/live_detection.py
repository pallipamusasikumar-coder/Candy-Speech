import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np
from joblib import load


# =====================================================
# SETTINGS
# =====================================================

MODEL_PATH = "sound_model.joblib"

SAMPLE_RATE = 22050
DURATION = 3


# =====================================================
# LOAD MODEL
# =====================================================

data = load(MODEL_PATH)

if isinstance(data, dict):

    model = data["model"]

    categories = data.get(
        "categories",
        ["human", "animal", "noise"]
    )

else:

    model = data

    categories = [
        "human",
        "animal",
        "noise"
    ]


# =====================================================
# FEATURE EXTRACTION
# =====================================================

def extract_features(audio, sr):

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13
    )

    mfcc_mean = np.mean(
        mfcc,
        axis=1
    )

    mfcc_std = np.std(
        mfcc,
        axis=1
    )


    # RMS Energy
    rms = librosa.feature.rms(
        y=audio
    )

    rms_mean = np.mean(rms)


    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(
        audio
    )

    zcr_mean = np.mean(zcr)


    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr
    )

    centroid_mean = np.mean(
        centroid
    )


    # Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sr
    )

    bandwidth_mean = np.mean(
        bandwidth
    )


    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr
    )

    rolloff_mean = np.mean(
        rolloff
    )


    # Chroma
    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr
    )

    chroma_mean = np.mean(
        chroma,
        axis=1
    )


    # Combine everything
    features = np.concatenate([

        mfcc_mean,

        mfcc_std,

        [rms_mean],

        [zcr_mean],

        [centroid_mean],

        [bandwidth_mean],

        [rolloff_mean],

        chroma_mean

    ])

    return features


# =====================================================
# START
# =====================================================

print("\n===================================")
print("🎙️ CANDYSOUND AI")
print("===================================")

print("\nAvailable Classes:")

for i, category in enumerate(categories):

    print(
        f"{i} → {category.upper()}"
    )


print(
    f"\n🎤 Speak/make a sound for {DURATION} seconds..."
)

print("Recording...\n")


# =====================================================
# RECORD AUDIO
# =====================================================

try:

    recording = sd.rec(

        int(
            DURATION *
            SAMPLE_RATE
        ),

        samplerate=SAMPLE_RATE,

        channels=1,

        dtype="float32"

    )

    sd.wait()


except Exception as e:

    print(
        "\n❌ Microphone error:"
    )

    print(e)

    print(
        "\nCheck Windows microphone permissions "
        "and your default recording device."
    )

    exit()


# =====================================================
# CONVERT AUDIO
# =====================================================

audio = recording.flatten()


# =====================================================
# SAVE AUDIO
# =====================================================

sf.write(

    "live_audio.wav",

    audio,

    SAMPLE_RATE

)


print(
    "✅ Audio recorded."
)

print(
    "🎵 Audio saved as live_audio.wav"
)


# =====================================================
# EXTRACT FEATURES
# =====================================================

try:

    features = extract_features(

        audio,

        SAMPLE_RATE

    )

except Exception as e:

    print(
        "\n❌ Feature extraction error:"
    )

    print(e)

    exit()


print(
    f"🧠 Feature shape: {features.shape}"
)


# =====================================================
# CHECK FEATURE SIZE
# =====================================================

expected_features = getattr(

    model,

    "n_features_in_",

    None

)


if expected_features is not None:

    if len(features) != expected_features:

        print("\n❌ Feature mismatch!")

        print(
            f"Model expects: {expected_features}"
        )

        print(
            f"Received: {len(features)}"
        )

        exit()


# =====================================================
# PREDICTION
# =====================================================

try:

    prediction = model.predict(

        [features]

    )[0]


    prediction = int(
        prediction
    )


except Exception as e:

    print(
        "\n❌ Prediction error:"
    )

    print(e)

    exit()


# =====================================================
# RESULT
# =====================================================

if prediction < len(categories):

    result = categories[
        prediction
    ]

else:

    result = str(
        prediction
    )


# =====================================================
# CONFIDENCE
# =====================================================

confidence = None

probabilities = None


if hasattr(

    model,

    "predict_proba"

):

    probabilities = model.predict_proba(

        [features]

    )[0]

    confidence = (

        np.max(
            probabilities
        ) * 100

    )


# =====================================================
# DISPLAY RESULT
# =====================================================

print("\n===================================")

print("🎯 PREDICTION RESULT")

print("===================================")


if result.lower() == "human":

    print(
        "👤 Prediction : HUMAN"
    )

elif result.lower() == "animal":

    print(
        "🐕 Prediction : ANIMAL"
    )

elif result.lower() == "noise":

    print(
        "🔊 Prediction : NOISE"
    )

else:

    print(
        f"🎵 Prediction : {result.upper()}"
    )


if confidence is not None:

    print(
        f"🎯 Confidence : {confidence:.2f}%"
    )


# =====================================================
# CLASS PROBABILITIES
# =====================================================

if probabilities is not None:

    print(
        "\n📊 Class Probabilities:"
    )

    for i, probability in enumerate(
        probabilities
    ):

        if i < len(categories):

            name = categories[i]

        else:

            name = f"Class {i}"


        print(

            f"{name.upper():10} : "
            f"{probability * 100:.2f}%"

        )


print("===================================\n")