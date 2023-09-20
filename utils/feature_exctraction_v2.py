import librosa
import numpy as np
from utils.audio_preprocess import random_augmentation
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import uuid
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
#import pylab
#from pylab import specgram, savefig, close


def extract_3dfeatures(data, sample_rate=22050, n_mfcc=13, feats=True):
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
        result = librosa.power_to_db(result)
        #if not os.path.isdir('Data/spectr'):
        #    fig = plt.figure(figsize=(10, 6))
        #    plt.imshow(result, aspect='auto', origin='lower', cmap='viridis')
        #    plt.colorbar(format='%+2.0f dB')
        #    plt.xlabel('Time')
        #    plt.ylabel('Frequency')
        #    plt.title('Spectrogram')
        #    filename = str(uuid.uuid4())
        #    dir = ('Data/spectr/')
        #    plt.savefig((dir + f'{filename}.png'), bbox_inches='tight')
        #    plt.close(fig)
        #    del fig

    return result


def get_3dfeatures(path, n_aug=0, feats=True):
    data, sample_rate = librosa.load(path)  # , duration=2.5, offset=0.6

    # without augmentation
    res1 = extract_3dfeatures(data, feats=feats)
    result = np.array(res1)[np.newaxis]

    synth_results = list()
    if n_aug != 0:
        synth_results.append(result[0])
        for _ in range(n_aug):
            # data with random augmentation
            noise_data = random_augmentation(data, sample_rate)
            res2 = extract_3dfeatures(noise_data, feats=feats)
            synth_results.append(res2)
        result = synth_results

    return result


def get3d_data(data_path, feats = False, max_aug = 0, save_png = False): # adjust n_aug for the number of synth data
    max_aug = max_aug + 1 # +1 because random goes from 0 to n - 1 
    if (save_png):
        x_train, y_train = [], []
        if not feats and not os.path.isdir('Data/spectr') and save_png:
            os.mkdir('Data/spectr')
            print('Directory "Data/spectr/" to store spectrograms created.')
        for path, emotion in zip(data_path.Path, data_path.Emotions):
            n_aug = np.random.randint(0, max_aug) 
            feature = get_3dfeatures(path, n_aug=n_aug, feats=feats)
            for ele in feature:
                spect_png(ele, emotion)
        print("Exctraction complete. x ->", len(x_train))
        return [], [], [], [], []
    else:
        train, test = train_test_split(data_path, random_state=42, shuffle=True)
        x_train, y_train = [], []
        x_test, y_test = [], []
        print("Exctracting and processing data...")
        
        # TRAIN + Augmentation
        for path, emotion in zip(train.Path, train.Emotions):
            n_aug = np.random.randint(0, max_aug) 
            feature = get_3dfeatures(path, n_aug=n_aug, feats=feats)
            for ele in feature:
                x_train.append(ele)
                y_train.append(emotion)
        print("Train exctraction complete. x_train ->", len(x_train))

        # TEST
        for path, emotion in zip(test.Path, test.Emotions):
            feature = get_3dfeatures(path, feats=feats)
            for ele in feature:
                x_test.append(ele)
                y_test.append(emotion)
        print("Test exctraction complete. x_test ->", len(x_test))

        print("Final processing for NN.")
        # Match rows and columns, because the time of the audios are different
        max_rows, max_cols = 0,0
        for atens in [x_train, x_test]: # Find the maximum number of rows and columns among all arrays
            max_rows = max(max_rows, max(a.shape[0] for a in atens))
            max_cols = max(max_cols, max(a.shape[1] for a in atens))

        for atens in [x_train, x_test]:
            for i, a in enumerate(atens): # Pad the smaller arrays with zeros to match the maximum shape
                atens[i] = np.pad(a, ((0, max_rows - a.shape[0]), (0, max_cols - a.shape[1])), 'constant')

        # Stack the padded arrays using numpy.stack()
        x_train = np.stack(x_train)
        x_test = np.stack(x_test)

        # one hot encoding
        encoder = OneHotEncoder()
        y_train = encoder.fit_transform(np.array(y_train).reshape(-1,1)).toarray()
        y_test = encoder.fit_transform(np.array(y_test).reshape(-1,1)).toarray()

        # making our data compatible to model.
        x_train = np.expand_dims(x_train, axis=-1)
        x_test =  np.expand_dims(x_test, axis=-1)
        print("x_train.shape -> ", x_train.shape)
        print("y_train.shape -> ", y_train.shape)
        print("x_test.shape -> ", x_test.shape)
        print("y_test.shape -> ", y_test.shape)
        print("Done.")

        return x_train, y_train, x_test, y_test, encoder

def spect_png(ele, emotion):
    if not os.path.isdir(f'Data/spectr/{emotion}'):
        os.mkdir(f'Data/spectr/{emotion}')
    fig = plt.figure(num=1, clear=True, figsize=(10, 6))
    plt.imshow(ele, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title('Spectrogram')
    filename = str(uuid.uuid4())
    plt.savefig((f'Data/spectr/{emotion}/{filename}.png'), bbox_inches='tight')

# Function to prepare our datasets for modelling
def prepare_png(ds, augment=False):
    # Define our one transformation
    rescale = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.Rescaling(1./255)])
    flip_and_rotate = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)
    ])
    
    # Apply rescale to both datasets and augmentation only to training
    ds = ds.map(lambda x, y: (rescale(x, training=True), y))
    if augment: ds = ds.map(lambda x, y: (flip_and_rotate(x, training=True), y))
    return ds

def png_processing(OUTPUT_DIR = 'Data/spectr/', BATCH_SIZE = 32, IMAGE_HEIGHT = 256, IMAGE_WIDTH = 256):

    # Make a dataset containing the training spectrograms
    xpng_train = tf.keras.preprocessing.image_dataset_from_directory(
                                                 batch_size=BATCH_SIZE,
                                                 validation_split=0.2,
                                                 directory=OUTPUT_DIR,
                                                 shuffle=True,
                                                 label_mode='categorical',
                                                 color_mode='rgb',
                                                 image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                                 subset="training",
                                                 seed=42)

    # Make a dataset containing the validation spectrogram
    xpng_test = tf.keras.preprocessing.image_dataset_from_directory(
                                                 batch_size=BATCH_SIZE,
                                                 validation_split=0.2,
                                                 directory=OUTPUT_DIR,
                                                 shuffle=True,
                                                 label_mode='categorical',
                                                 color_mode='rgb',
                                                 image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                                 subset="validation",
                                                 seed=42)

    xpng_train = prepare_png(xpng_train, augment=False)
    xpng_test = prepare_png(xpng_test, augment=False)
    return xpng_train, xpng_test