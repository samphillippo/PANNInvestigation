import sys
import os
import pandas as pd
import librosa
import numpy as np
import torch

from FeatureExtractor import get_features_from_wav, get_label_vector
from ResNet38 import ResNet38_Transfer
from DataSet import GTZANDataset
from TrainModel import train

sample_rate = 32000
clip_duration = 30
max_len = sample_rate * clip_duration
window_size = 1024
hop_size=320
mel_bins=64
fmin=50
fmax=14000
classes_num = 10

genre_to_index_map = { "blues": 0, "classical": 1, "country": 2, "disco": 3, "hiphop": 4, "jazz": 5, "metal": 6, "pop": 7, "reggae": 8, "rock": 9 }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python FineTuneGTZAN.py <path_to_GTZAN_dataset> <path_to_pretrained_model>")
        sys.exit(1)


    model = ResNet38_Transfer(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num)
    model.load_from_pretrain(sys.argv[2])

    datasetPath = sys.argv[1]
    data = []
    count = 1
    for genre_folder in os.listdir(datasetPath):
        genre_path = os.path.join(datasetPath, genre_folder)
        if os.path.isdir(genre_path):
            for filename in os.listdir(genre_path):
                if filename.endswith(".wav"):
                    file_path = os.path.join(genre_path, filename)
                    data.append({"audio": get_features_from_wav(file_path, sample_rate, max_len), "genre": get_label_vector(genre_folder, genre_to_index_map), "fold": count % 10})
                    count += 1

    dataset = GTZANDataset(data)
    train(model, dataset)
