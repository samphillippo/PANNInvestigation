import sys
import os
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def measure_confusion(model, dataset, workspace, dataset_name):
    # confusion_matrix = np.zeros((classes_num, classes_num), dtype=np.int)

    # for i in range(len(dataset)):
    #     waveform = dataset[i]["waveform"]
    #     target = dataset[i]["target"]

    #     waveform = waveform.unsqueeze(0)
    #     waveform = waveform.cuda()
    #     model = model.cuda()
    #     model.eval()

    #     with torch.no_grad():
    #         output = model(waveform)
    #         output = torch.sigmoid(output)

    #     output = output.cpu().numpy().flatten()
    #     target = target.numpy().flatten()

    #     predicted = np.argmax(output)
    #     actual = np.argmax(target)

    predicted_labels = np.array([0, 1, 1, 2, 2, 0])
    ground_truth_labels = np.array([0, 1, 2, 2, 1, 0])

    return confusion_matrix(ground_truth_labels, predicted_labels)

def plot_confusion(conf_matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g',
                xticklabels=genre_to_index_map.keys(),
                yticklabels=genre_to_index_map.keys())
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python MeasureConfusion.py <workspace_path> <GTZAN_dataset_name> <model_name>")
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
    dataset = load_gtzan_dataset(datasetPath, sample_rate, max_len)
    print("Measuring confusion matrix...")

    conf_matrix = measure_confusion(model, dataset, workspace, "GTZAN")
    plot_confusion(conf_matrix)
