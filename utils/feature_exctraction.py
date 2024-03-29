import librosa
import numpy as np
import pandas as pd
from utils.audio_preprocess import random_augmentation
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
import uuid
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split


def extract_features(data, sample_rate = 22050, n_mfcc = 13, feats = True):
    if feats:
        # ZCR
        result = np.array([])
        zcr = librosa.feature.zero_crossing_rate(y=data)
        result = np.hstack((result, np.mean(zcr, axis=1))) # MEAN - stacking horizontally
        result = np.hstack((result, np.std(zcr, axis=1))) # STD - stacking horizontally

        # Chroma_stft
        #stft = np.abs(librosa.stft(data))
        #chroma_stft = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
        #result = np.hstack((result, np.mean(chroma_stft, axis = 1))) # MEAN - stacking horizontally
        #result = np.hstack((result, np.std(chroma_stft, axis=1))) # STD - stacking horizontally

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

        # MelSpectogram std
        mel = librosa.feature.melspectrogram(y=data, sr=sample_rate)
        result = np.hstack((result, np.mean(mel,  axis=1))) # MEAN - stacking horizontally
        result = np.hstack((result, np.std(mel, axis=1))) # STD - stacking horizontally

    else:
        # MelSpectogram
        result = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)

    return result

def get_features(path, n_aug = 0, feats = True, debug = False):
    data, sample_rate = librosa.load(path) # , duration=2.5, offset=0.6
    
    # without augmentation
    res1 = extract_features(data, feats = feats)
    if (debug == True):
        print("pre new axis:", res1.shape)

    if (n_aug > 0):
        result = np.array(res1)
        counter = 0
        while counter < n_aug:
            # data with random augmentation
            noise_data = random_augmentation(data, sample_rate)
            res2 = extract_features(noise_data, feats = feats)
            result = np.vstack((result, res2)) # stacking vertically
            counter += 1
    else:
        result = np.array(res1)[np.newaxis]
        if (debug == True):
            print("post new axis:", result.shape)
    
    return result

def get2d_data (data_path, max_aug = 1, feats = True, save_png = False):
    max_aug = max_aug + 1 # +1 because random goes from 0 to n - 1 
    if (save_png):
        #x_train, y_train = [], []
        #if not feats and not os.path.isdir('Data/spectr') and save_png:
        #    os.mkdir('Data/spectr')
        #    print('Directory "Data/spectr/" to store spectrograms created.')
        #for path, emotion in zip(data_path.Path, data_path.Emotions):
        #    n_aug = np.random.randint(0, max_aug) 
        #    feature = get_3dfeatures(path, n_aug=n_aug, feats=feats)
        #    for ele in feature:
        #        spect_png(ele, emotion)
        #print("Exctraction complete. x ->", len(x_train))
        return [], [], [], []
    else:
        train, test = train_test_split(data_path, random_state=42, shuffle=True)
        x_train, y_train = [], []
        x_test, y_test = [], []
        print("Exctracting and processing data...")
        
        # TRAIN + Augmentation
        for path, emotion in zip(train.Path, train.Emotions):
            n_aug = np.random.randint(0, max_aug) 
            feature = get_features(path, n_aug=n_aug, feats=feats)
            for ele in feature:
                x_train.append(ele)
                y_train.append(emotion)
        print("Train exctraction complete. x_train ->", len(x_train))

        # TEST
        for path, emotion in zip(test.Path, test.Emotions):
            feature = get_features(path, feats=feats)
            for ele in feature:
                x_test.append(ele)
                y_test.append(emotion)
        print("Test exctraction complete. x_test ->", len(x_test))

        print("Final processing for NN.")
    
    # I want them as arrays
    Matrix_train = pd.DataFrame(x_train)
    Matrix_train['labels'] = y_train
    Matrix_test = pd.DataFrame(x_test)
    Matrix_test['labels'] = y_test
    x_train = Matrix_train.iloc[: ,:-1].values
    y_train = Matrix_train['labels'].values
    x_test = Matrix_test.iloc[: ,:-1].values
    y_test = Matrix_test['labels'].values
    
    # one hot encoding
    encoder = OneHotEncoder()
    y_train = encoder.fit_transform(np.array(y_train).reshape(-1,1)).toarray()
    y_test = encoder.fit_transform(np.array(y_test).reshape(-1,1)).toarray()
    
    # standardScaler    
    #scaler = StandardScaler()
    #for i in range(x_train.shape[0]):
    #    x_train[i] = scaler.fit_transform(x_train[i])


    #for i in range(x_test.shape[0]):
    #    x_test[i] = scaler.fit_transform(x_test[i])
            
    return x_train, y_train, x_test, y_test, encoder