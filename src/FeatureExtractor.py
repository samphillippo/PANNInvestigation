import librosa
import numpy as np
import torch
import os

# Extracts features from a wav file
def get_features_from_wav(file_path, sample_rate=32000, max_len=960000):
    audio, fs = librosa.core.load(file_path, sr=sample_rate, mono=True)

    if len(audio) < max_len:
        # Pad audio file if it is less than 30 seconds
        audio = np.concatenate((audio, np.zeros(max_len - len(audio))))
    else:
        # Cut audio file if it is more than 30 seconds
        audio = audio[0 : max_len]

    #TODO: do we still want to normalize the audio?
    if np.max(np.abs(audio)) > 1.:
        audio /= np.max(np.abs(audio))

    return torch.Tensor(audio)

# Gets a one-hot encoded label vector for the genre
def get_label_vector(genre_label):
    genre_to_index_map = { "blues": 0, "classical": 1, "country": 2, "disco": 3, "hiphop": 4, "jazz": 5, "metal": 6, "pop": 7, "reggae": 8, "rock": 9 }
    label_vector = np.zeros(len(genre_to_index_map), dtype=np.float32)
    label_vector[genre_to_index_map[genre_label]] = 1
    return torch.Tensor(label_vector)

def load_gtzan_dataset(filepath, sample_rate, max_len):
    data = []
    count = 1
    for genre_folder in os.listdir(filepath):
        genre_path = os.path.join(filepath, genre_folder)
        if os.path.isdir(genre_path):
            for filename in os.listdir(genre_path):
                if filename.endswith(".wav"):
                    file_path = os.path.join(genre_path, filename)
                    #waveform = get_features_from_wav(file_path, sample_rate, max_len).unsqueeze(0)
                    waveform = get_features_from_wav(file_path, sample_rate, max_len)
                    data.append({"filename": filename, "waveform": waveform, "target": get_label_vector(genre_folder), "fold": count % 10})
                    #print(data[-1]["waveform"].shape)
                    count += 1

    return data
