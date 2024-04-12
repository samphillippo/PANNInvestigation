import torch
import torch.nn as nn

class ResNet38(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
        fmax, classes_num):

        super(ResNet38, self).__init__()

        #need spectrogram extractor
        #need logmel extractor
        # need "spec augmenter"

        #THEN we add actual network structure



#TODO: add final layer for transfer task
class ResNet38_Transfer(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
        fmax, classes_num, freeze_base):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(ResNet38, self).__init__()

        audioset_classes_num = 527

        self.base = ResNet38(sample_rate, window_size, hop_size, mel_bins, fmin,
            fmax, audioset_classes_num)

        # transfer lyaer:
        #self.fc_transfer = nn.Linear(2048, classes_num, bias=True)

    #need load from pretrain also!!!
    #this should match exactly structure!!!
