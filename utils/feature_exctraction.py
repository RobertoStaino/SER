import librosa
import numpy as np
from utils.audio_preprocess import random_augmentation

def extract_features(data, sample_rate = 22050, n_mfcc = 13):
    # ZCR
    result = np.array([])
    zcr = librosa.feature.zero_crossing_rate(y=data)
    result = np.hstack((result, np.mean(zcr, axis=1))) # MEAN - stacking horizontally
    result = np.hstack((result, np.std(zcr, axis=1))) # STD - stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
    result = np.hstack((result, np.mean(chroma_stft, axis = 1))) # MEAN - stacking horizontally
    result = np.hstack((result, np.std(chroma_stft, axis=1))) # STD - stacking horizontally

    # MFCC
    mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc = n_mfcc)
    result = np.hstack((result, np.mean(mfcc, axis=1))) # MEAN - stacking horizontally
    result = np.hstack((result, np.std(mfcc, axis=1))) # STD - stacking horizontally

    # spectral centroids 
    x_spectral_centroid = librosa.feature.spectral_centroid(y=data, sr=sample_rate)
    result = np.hstack((result, x_spectral_centroid.mean())) # MEAN - stacking horizontally
    result = np.hstack((result, x_spectral_centroid.std())) # STD - stacking horizontally
    x_spectral_bw = librosa.feature.spectral_bandwidth(y=data, sr=sample_rate)
    result = np.hstack((result, x_spectral_bw.mean())) # MEAN - tacking horizontally
    result = np.hstack((result, x_spectral_bw.std())) # STD - stacking horizontally

    # Root Mean Square Value
    rms = librosa.feature.rms(y=data)
    result = np.hstack((result, np.mean(rms,  axis=1))) # MEAN - stacking horizontally
    result = np.hstack((result, np.std(rms, axis=1))) # STD - stacking horizontally

    # MelSpectogram
    mel = librosa.feature.melspectrogram(y=data, sr=sample_rate)
    #result = np.hstack((result, np.mean(mel,  axis=1))) # MEAN - stacking horizontally
    result = np.hstack((result, np.std(mel, axis=1))) # STD - stacking horizontally
    
    return result

def get_features(path, synth = True, n_iter = 1):
    data, sample_rate = librosa.load(path) # , duration=2.5, offset=0.6
    
    # without augmentation
    res1 = extract_features(data)

    if (synth):
        result = np.array(res1)
        counter = 0
        while counter < n_iter:
            # data with random augmentation
            noise_data = random_augmentation(data, sample_rate)
            res2 = extract_features(noise_data)
            result = np.vstack((result, res2)) # stacking vertically
            counter += 1
    else:
        result = np.array(res1)[np.newaxis]
    
    return result