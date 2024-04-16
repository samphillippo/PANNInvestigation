import sys
import os

from FeatureExtractor import get_features_from_wav, get_label_vector
from ResNet38 import ResNet38_Transfer
from CNN14 import Transfer_Cnn14
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

# Fine-tune the model on the GTZAN dataset
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python FineTuneGTZAN.py <workspace_path> <GTZAN_dataset_name> <pretrained_model_name>")
        sys.exit(1)

    workspace = sys.argv[1]

    print("Loading model...")
    modelName = sys.argv[3]
    if "ResNet38" in modelName:
        model = ResNet38_Transfer(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num)
    elif "Cnn14" in modelName:
        model = Transfer_Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, False)
    else:
        print("Invalid model type")
        sys.exit(1)
    model.load_from_pretrain(os.path.join(workspace, modelName))

    print("Loading dataset...")
    datasetPath = os.path.join(workspace, sys.argv[2])
    data = []
    count = 1
    for genre_folder in os.listdir(datasetPath):
        genre_path = os.path.join(datasetPath, genre_folder)
        if os.path.isdir(genre_path):
            for filename in os.listdir(genre_path):
                if filename.endswith(".wav"):
                    file_path = os.path.join(genre_path, filename)
                    #waveform = get_features_from_wav(file_path, sample_rate, max_len).unsqueeze(0)
                    waveform = get_features_from_wav(file_path, sample_rate, max_len)
                    data.append({"filename": filename, "waveform": waveform, "target": get_label_vector(genre_folder, genre_to_index_map), "fold": count % 10})
                    #print(data[-1]["waveform"].shape)
                    count += 1

    dataset = GTZANDataset(data)
    print("Fine-tuning model...")
    train(model, dataset, workspace, "GTZAN")
