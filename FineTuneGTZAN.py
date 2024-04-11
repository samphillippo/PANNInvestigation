import sys
import os
import pandas as pd
import librosa
import numpy as np

sample_rate = 32000
clip_duration = 30
max_len = sample_rate * clip_duration

genre_to_index_map = { "blues": 0, "classical": 1, "country": 2, "disco": 3, "hiphop": 4, "jazz": 5, "metal": 6, "pop": 7, "reggae": 8, "rock": 9 }

def FineTuneGTZAN(datasetPath):
    data = []
    for genre_folder in os.listdir(datasetPath):
        genre_path = os.path.join(datasetPath, genre_folder)
        if os.path.isdir(genre_path):
            for filename in os.listdir(genre_path):
                if filename.endswith(".wav"):
                    file_path = os.path.join(genre_path, filename)
                    genre_label = genre_folder

                    #need sample rate
                    audio, fs = librosa.core.load(file_path, sr=sample_rate, mono=True)

                    if len(audio) < max_len:
                        # Pad audio file if it is less than 30 seconds
                        audio = np.concatenate((audio, np.zeros(max_len - len(audio))))
                    else:
                        # Cut audio file if it is more than 30 seconds
                        audio = audio[0 : max_len]

                    # Convert audio to 16 bit integer
                    if np.max(np.abs(audio)) > 1.:
                        audio /= np.max(np.abs(audio))
                    audio = (audio * 32767.).astype(np.int16)

                    label_vector = np.zeros(10)
                    label_vector[genre_to_index_map[genre_label]] = 1
                    data.append({"audio": audio, "genre": label_vector})


    return pd.DataFrame(data)
