import librosa
import numpy as np


def extract_features(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=22050,
        mono=True
    )

    # -----------------------------
    # MFCC
    # -----------------------------

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

    # -----------------------------
    # RMS ENERGY
    # -----------------------------

    rms = librosa.feature.rms(
        y=audio
    )

    rms_mean = np.mean(rms)

    # -----------------------------
    # ZERO CROSSING RATE
    # -----------------------------

    zcr = librosa.feature.zero_crossing_rate(
        audio
    )

    zcr_mean = np.mean(zcr)

    # -----------------------------
    # SPECTRAL CENTROID
    # -----------------------------

    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr
    )

    centroid_mean = np.mean(
        spectral_centroid
    )

    # -----------------------------
    # SPECTRAL BANDWIDTH
    # -----------------------------

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sr
    )

    bandwidth_mean = np.mean(
        spectral_bandwidth
    )

    # -----------------------------
    # SPECTRAL ROLLOFF
    # -----------------------------

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr
    )

    rolloff_mean = np.mean(
        spectral_rolloff
    )

    # -----------------------------
    # CHROMA
    # -----------------------------

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr
    )

    chroma_mean = np.mean(
        chroma,
        axis=1
    )

    # -----------------------------
    # COMBINE FEATURES
    # -----------------------------

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