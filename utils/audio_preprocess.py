import librosa
import numpy as np
import os
import pandas as pd

# each audio file is divided into samples, expressed by frequency. There are sr samples in a seconds.
# in the spectrogram, the fast fourier transormations shrink the number of samples.
# in this case the n_ftt is 512 so a second is given by 512/4 units.
# corresponding to 23 milliseconds at a sample rate of 22050 Hz.

def unit_to_seconds(unit, sr = 22050):
    return int(unit / sr)

def second_to_unit(second, sr = 22050):
    return int(second * sr)

def uAudio_to_uSpect(unit, n_ftt = 512):
    return int(unit / (n_ftt // 4) + 1)

def unify_intervals(intervals):
    # Sort the intervals by the start time
    intervals = sorted(intervals, key=lambda x: x[0])
    
    # Merge overlapping intervals
    merged_intervals = []
    for interval in intervals:
        if not merged_intervals or merged_intervals[-1][1] < interval[0]:
            # If the current interval does not overlap with the previous interval, add it to the list
            merged_intervals.append(interval)
        else:
            # If the current interval overlaps with the previous interval, merge them
            merged_intervals[-1] = (merged_intervals[-1][0], max(merged_intervals[-1][1], interval[1]))
    
    return merged_intervals

# this functions identifies low decibel segments and returns only speeched frames
def audio_section(Xdb, sr = 22050, window = 0.5, threshold = -40): # works only if n_ftt = 512
    Xdb_mean = np.mean(Xdb, axis=0)

    # Define the size of the window --- (sr / 512 // 4) is a second
    n = round((sr / (512 // 4))*window)

    arr = np.pad(Xdb_mean, (0, n - len(Xdb_mean) % n), mode='constant')

    # Compute the mean of every n elements and shrink the array length
    mean_arr = np.mean(arr[:len(arr)//n*n].reshape(-1, n), axis=1)
    
    start_time = 0
    end_time = window

    speech_sections = []
    for i in range(mean_arr.shape[0]): #each i is "window" seconds
        if mean_arr[i] > threshold:
            # Add the start ad end time of the low energy section to the list
            speech_sections.append((start_time, end_time))
        start_time += window
        end_time += window

    # # Extract the non-speech segments from the audio file
    # audio_sections = np.empty(0, dtype=np.float32)

    # for segment in unify_intervals(speech_sections):
    #     start_time = second_to_unit(segment[0])
    #     end_time = second_to_unit(segment[1])
    #     audio_sections.append(x[start_time:end_time])

    return unify_intervals(speech_sections) 

### Data augmentation function

def random_augmentation(data, sr = 22050, noise = 0.005):

    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    rate=np.random.uniform(0.6, 1.4)
    pitch_factor=np.random.uniform(0.5, 1.5)
    noise_amp = noise*np.random.uniform()*np.amax(data)

    # add random if ?
    data = np.roll(data, shift_range) # shifting
    data = librosa.effects.time_stretch(data, rate=rate) # stretching
    data = librosa.effects.pitch_shift(data, sr=sr, n_steps=pitch_factor) # pitching
    data = data + noise_amp*np.random.normal(size=data.shape[0]) # noise injection

    return data

def data_path(save_csv = False):
    Crema = "CREMA-D/AudioWAV/"
    Ravdess = "RAVDESS/audio_speech_actors_01-24/"
    Tess = "TESS/"
    Savee = "Savee/"
    crema_directory_list = os.listdir(Crema)
    ravdess_directory_list = os.listdir(Ravdess)
    tess_directory_list = os.listdir(Tess)
    savee_directory_list = os.listdir(Savee)
    
    # RADVESS
    file_emotion = []
    file_path = []
    for dir in ravdess_directory_list:
        if not dir.startswith('.'):
            actor = os.listdir(Ravdess + dir)
            for file in actor:
                if not file.startswith('.'):
                    part = file.split('.')[0]
                    part = part.split('-')
                    file_emotion.append(int(part[2]))
                    file_path.append(Ravdess + dir + '/' + file)

    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    Ravdess_df = pd.concat([emotion_df, path_df], axis=1)
    Ravdess_df.Emotions.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)

    #Crema
    file_emotion = []
    file_path = []
    for file in crema_directory_list:
        if not file.startswith('.'):
            file_path.append(Crema + file)
            part=file.split('_')
            if part[2] == 'SAD':
                file_emotion.append('sad')
            elif part[2] == 'ANG':
                file_emotion.append('angry')
            elif part[2] == 'DIS':
                file_emotion.append('disgust')
            elif part[2] == 'FEA':
                file_emotion.append('fear')
            elif part[2] == 'HAP':
                file_emotion.append('happy')
            elif part[2] == 'NEU':
                file_emotion.append('neutral')
            else:
                file_emotion.append('Unknown')
            
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    Crema_df = pd.concat([emotion_df, path_df], axis=1)
    Crema_df['Emotions'] = pd.Categorical(Crema_df['Emotions'])
    
    # TESS
    file_emotion = []
    file_path = []
    for dir in tess_directory_list:
        if not dir.startswith('.'):
            directories = os.listdir(Tess + dir)
            for file in directories:
                if not file.startswith('.'):
                    part = file.split('.')[0]
                    part = part.split('_')[2]
                    if part=='ps':
                        file_emotion.append('surprise')
                    else:
                        file_emotion.append(part)
                    file_path.append(Tess + dir + '/' + file)
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    Tess_df = pd.concat([emotion_df, path_df], axis=1)
    
    # Savee
    file_emotion = []
    file_path = []
    for file in savee_directory_list:
        if not file.startswith('.'):
            file_path.append(Savee + file)
            part = file.split('_')[1]
            ele = part[:-6]
            if ele=='a':
                file_emotion.append('angry')
            elif ele=='d':
                file_emotion.append('disgust')
            elif ele=='f':
                file_emotion.append('fear')
            elif ele=='h':
                file_emotion.append('happy')
            elif ele=='n':
                file_emotion.append('neutral')
            elif ele=='sa':
                file_emotion.append('sad')
            else:
                file_emotion.append('surprise')
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    Savee_df = pd.concat([emotion_df, path_df], axis=1)
    
    data_path = pd.concat([Ravdess_df, Crema_df, Tess_df, Savee_df], axis = 0)
    if (save_csv): data_path.to_csv("Data/data_path.csv",index=False)
    
    # remove unbalanced
    data_path = data_path[data_path.Emotions != "calm"]
    data_path = data_path[data_path.Emotions != "surprise"]
    
    return data_path