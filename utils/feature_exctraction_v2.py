import librosa
import numpy as np
from utils.audio_preprocess import random_augmentation


def extract_features(data, sample_rate=22050, n_mfcc=13, spect=False):
    if spect == False:
        # ZCR
        result = np.array([])
        zcr = librosa.feature.zero_crossing_rate(y=data)
        result = np.hstack(
            (result, np.mean(zcr, axis=1))
        )  # MEAN - stacking horizontally
        result = np.hstack((result, np.std(zcr, axis=1)))  # STD - stacking horizontally

        # Chroma_stft
        # stft = np.abs(librosa.stft(data))
        # chroma_stft = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
        # result = np.hstack((result, np.mean(chroma_stft, axis = 1))) # MEAN - stacking horizontally
        # result = np.hstack((result, np.std(chroma_stft, axis=1))) # STD - stacking horizontally

        # MFCC
        mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=n_mfcc)
        result = np.hstack(
            (result, np.mean(mfcc, axis=1))
        )  # MEAN - stacking horizontally
        result = np.hstack(
            (result, np.std(mfcc, axis=1))
        )  # STD - stacking horizontally

        # spectral centroids
        x_spectral_centroid = librosa.feature.spectral_centroid(y=data, sr=sample_rate)
        result = np.hstack(
            (result, x_spectral_centroid.mean())
        )  # MEAN - stacking horizontally
        result = np.hstack(
            (result, x_spectral_centroid.std())
        )  # STD - stacking horizontally
        x_spectral_bw = librosa.feature.spectral_bandwidth(y=data, sr=sample_rate)
        result = np.hstack(
            (result, x_spectral_bw.mean())
        )  # MEAN - tacking horizontally
        result = np.hstack((result, x_spectral_bw.std()))  # STD - stacking horizontally

        # Root Mean Square Value
        rms = librosa.feature.rms(y=data)
        result = np.hstack(
            (result, np.mean(rms, axis=1))
        )  # MEAN - stacking horizontally
        result = np.hstack((result, np.std(rms, axis=1)))  # STD - stacking horizontally

        # MelSpectogram
        mel = librosa.feature.melspectrogram(y=data, sr=sample_rate)
        # result = np.hstack((result, np.mean(mel,  axis=1))) # MEAN - stacking horizontally
        result = np.hstack((result, np.std(mel, axis=1)))  # STD - stacking horizontally

    else:
        # MelSpectogram
        result = librosa.feature.melspectrogram(y=data, sr=sample_rate)

    return result


def get_features(path, n_aug=0, spect=False, debug=False):
    data, sample_rate = librosa.load(path)  # , duration=2.5, offset=0.6

    # without augmentation
    res1 = extract_features(data, spect=spect)
    result = np.array(res1)[np.newaxis]

    synth_results = list()
    if n_aug:
        synth_results.append(result[0])
        for _ in range(n_aug):
            # data with random augmentation
            noise_data = random_augmentation(data, sample_rate)
            res2 = extract_features(noise_data, spect=spect)
            synth_results.append(res2)
        result = synth_results

    return result
