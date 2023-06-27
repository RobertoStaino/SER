import librosa
import numpy as np


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