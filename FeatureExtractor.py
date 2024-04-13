import librosa
import numpy as np
import torch

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

# Gets a one-hot encoded label vector for the genre
def get_label_vector(genre_label, genre_to_index_map):
    label_vector = np.zeros(len(genre_to_index_map), dtype=np.float32)
    label_vector[genre_to_index_map[genre_label]] = 1
    return torch.Tensor(label_vector)
