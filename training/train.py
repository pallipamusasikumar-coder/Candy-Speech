import os
import librosa
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump


# =========================================================
# SETTINGS
# =========================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "sound_model.joblib"
)

SAMPLE_RATE = 22050
N_MFCC = 13

CATEGORIES = [
    "human",
    "animal",
    "noise"
]


# =========================================================
# FEATURE EXTRACTION
# EXACTLY 78 FEATURES
# =========================================================

def extract_features(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("Empty audio file")

    # -----------------------------------------
    # MFCC
    # -----------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    # -----------------------------------------
    # Delta
    # -----------------------------------------

    delta = librosa.feature.delta(
        mfcc
    )

    # -----------------------------------------
    # Delta Delta
    # -----------------------------------------

    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    # -----------------------------------------
    # 78 FEATURES
    # -----------------------------------------

    features = np.concatenate([

        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),

        np.mean(delta, axis=1),
        np.std(delta, axis=1),

        np.mean(delta2, axis=1),
        np.std(delta2, axis=1)

    ])

    return np.asarray(
        features,
        dtype=np.float64
    )


# =========================================================
# LOAD DATASET
# =========================================================

X = []
y = []

print("\nLoading dataset...\n")


for label, category in enumerate(CATEGORIES):

    folder = os.path.join(
        DATASET_PATH,
        category
    )

    if not os.path.exists(folder):

        print(
            f"WARNING: Folder not found: {folder}"
        )

        continue

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(".wav")
    ]

    print(
        f"{category}: {len(files)} files"
    )

    for file in files:

        file_path = os.path.join(
            folder,
            file
        )

        try:

            features = extract_features(
                file_path
            )

            if len(features) != 78:

                print(
                    f"Skipping {file}: "
                    f"{len(features)} features"
                )

                continue

            X.append(features)
            y.append(label)

        except Exception as e:

            print(
                f"Error processing {file}: {e}"
            )


# =========================================================
# CONVERT TO NUMPY
# =========================================================

X = np.asarray(
    X,
    dtype=np.float64
)

y = np.asarray(
    y,
    dtype=np.int32
)


print("\n===================================")
print("DATASET INFORMATION")
print("===================================")

print(
    "Feature Shape:",
    X.shape
)

print(
    "Label Shape:",
    y.shape
)

print(
    "Total Samples:",
    len(X)
)

print(
    "Number of Features:",
    X.shape[1]
    if len(X) > 0
    else 0
)


# =========================================================
# CHECK DATASET
# =========================================================

if len(X) == 0:

    raise RuntimeError(
        "No audio files were successfully processed."
    )


if X.shape[1] != 78:

    raise RuntimeError(
        f"Expected 78 features but got {X.shape[1]}"
    )


# =========================================================
# CLASS DISTRIBUTION
# =========================================================

print("\nClass Distribution:")

for index, category in enumerate(CATEGORIES):

    count = np.sum(
        y == index
    )

    print(
        f"{category}: {count}"
    )


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# =========================================================
# RANDOM FOREST
# =========================================================

model = RandomForestClassifier(

    n_estimators=200,

    max_depth=None,

    random_state=42,

    class_weight="balanced",

    n_jobs=-1

)


print("\nTraining Random Forest...\n")


model.fit(
    X_train,
    y_train
)


# =========================================================
# TEST
# =========================================================

y_pred = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n===================================")
print("MODEL PERFORMANCE")
print("===================================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


print("\nClassification Report:\n")


print(
    classification_report(
        y_test,
        y_pred,
        target_names=CATEGORIES
    )
)


# =========================================================
# SAVE MODEL
# =========================================================

model_data = {

    "model": model,

    "categories": CATEGORIES,

    "accuracy": accuracy,

    "sample_rate": SAMPLE_RATE,

    "n_mfcc": N_MFCC,

    "feature_count": 78

}


dump(
    model_data,
    MODEL_PATH
)


print("\n===================================")
print("MODEL SAVED")
print("===================================")

print(
    "Location:",
    MODEL_PATH
)

print(
    "Features: 78"
)

print(
    "Classes:",
    CATEGORIES
)

print(
    "Accuracy:",
    f"{accuracy * 100:.2f}%"
)