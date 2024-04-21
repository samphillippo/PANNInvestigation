import sys
import os

from FeatureExtractor import load_gtzan_dataset
from ResNet38 import ResNet38_Transfer
from CNN14 import Transfer_Cnn14
from TrainModel import train
from DataSet import GTZANDataset

sample_rate = 32000
clip_duration = 30
max_len = sample_rate * clip_duration
window_size = 1024
hop_size=320
mel_bins=64
fmin=50
fmax=14000
classes_num = 10

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
    dataset = GTZANDataset(load_gtzan_dataset(datasetPath, sample_rate, max_len))
    print("Fine-tuning model...")
    train(model, dataset, workspace, "GTZAN")
