import sys
import os
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from FeatureExtractor import load_gtzan_dataset
from ResNet38 import ResNet38_Transfer
from CNN14 import Transfer_Cnn14

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

def measure_confusion(model, dataset):
    device = 'cuda' if (torch.cuda.is_available()) else 'cpu'
    predicted = []
    actual = []

    print("Dataset length: ", len(dataset))
    for i in range(len(dataset)):
        if dataset[i]["fold"] != 1:
            continue
        waveform = dataset[i]["waveform"]
        target = dataset[i]["target"]

        waveform = waveform.unsqueeze(0)
        waveform = waveform.to(device)
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            output = model(waveform)["clipwise_output"]

        output = output.cpu().numpy().flatten()
        target = target.numpy().flatten()

        predicted.append(np.argmax(output))
        actual.append(np.argmax(target))
        print("Predicted: ", np.argmax(output), " Actual: ", np.argmax(target))

    return confusion_matrix(np.array(actual), np.array(predicted))

def plot_confusion(conf_matrix, model_type):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g',
                xticklabels=genre_to_index_map.keys(),
                yticklabels=genre_to_index_map.keys())
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix for ' + model_type + ' on GTZAN dataset')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python MeasureConfusion.py <workspace_path> <GTZAN_dataset_name> <model_name>")
        sys.exit(1)

    workspace = sys.argv[1]

    print("Loading model...")
    modelName = sys.argv[3]
    modelType = None
    if "ResNet38" in modelName:
        model = ResNet38_Transfer(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num)
        modelType = "ResNet38"
    elif "Cnn14" in modelName:
        model = Transfer_Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, False)
        modelType = "Cnn14"
    else:
        print("Invalid model type")
        sys.exit(1)
    model.load_from_pretrain(os.path.join(workspace, modelName))

    print("Loading dataset...")
    datasetPath = os.path.join(workspace, sys.argv[2])
    dataset = load_gtzan_dataset(datasetPath, sample_rate, max_len)
    print("Measuring confusion matrix...")

    conf_matrix = measure_confusion(model, dataset)
    plot_confusion(conf_matrix, modelType)
