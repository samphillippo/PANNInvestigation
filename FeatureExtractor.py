import librosa
import numpy as np
import torch

def get_features_from_wav(file_path, sample_rate=32000, max_len=960000):
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
    audio_tensor = torch.LongTensor(audio)
    return audio_tensor


#MAKE TENSOR
def get_label_vector(genre_label, genre_to_index_map):
    label_vector = np.zeros(len(genre_to_index_map))
    label_vector[genre_to_index_map[genre_label]] = 1
    return torch.Tensor(label_vector)
