import librosa
import numpy as np
from utils.audio_preprocess import random_augmentation


def extract_features(data, sample_rate=22050, n_mfcc=13, feats=True):
    if feats:
        # ZCR
        zcr = librosa.feature.zero_crossing_rate(y=data)
        # result = np.vstack((result, zcr))
        result = zcr

        # Chroma_stft
        # stft = np.abs(librosa.stft(data))
        # chroma_stft = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
        # result = np.vstack((result, chroma_stft))

        # MFCC
        mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=n_mfcc)
        result = np.vstack((result, mfcc))

        # spectral centroids
        x_spectral_centroid = librosa.feature.spectral_centroid(y=data, sr=sample_rate)
        result = np.vstack((result, x_spectral_centroid))

        # spectral bandwidth
        x_spectral_bw = librosa.feature.spectral_bandwidth(y=data, sr=sample_rate)
        result = np.vstack((result, x_spectral_bw))

        # Root Mean Square Value
        rms = librosa.feature.rms(y=data)
        result = np.vstack((result, rms))
    else:
        result = librosa.feature.melspectrogram(y=data, sr=sample_rate)

    return result


def get_features(path, n_aug=0, feats=True):
    data, sample_rate = librosa.load(path)  # , duration=2.5, offset=0.6

    # without augmentation
    res1 = extract_features(data, feats=feats)
    result = np.array(res1)[np.newaxis]

    synth_results = list()
    if n_aug:
        synth_results.append(result[0])
        for _ in range(n_aug):
            # data with random augmentation
            noise_data = random_augmentation(data, sample_rate)
            res2 = extract_features(noise_data, feats=feats)
            synth_results.append(res2)
        result = synth_results

    return result
