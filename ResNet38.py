import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=(3, 3), stride=(1, 1),
                            padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=(3, 3), stride=(1, 1),
                            padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        #NOTE: we don't need to init, always loading from pretrain

    def forward(self, input, pool_size=(2, 2)):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=pool_size)

        return x


class _ResnetBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(_ResnetBasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.stride = stride

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                     padding=1, groups=1, bias=False, dilation=1)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                     padding=1, groups=1, bias=False, dilation=1)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        #self.init_weights()

    # def init_weights(self):
    #     init_layer(self.conv1)
    #     init_bn(self.bn1)
    #     init_layer(self.conv2)
    #     init_bn(self.bn2)
    #     nn.init.constant_(self.bn2.weight, 0)

    def forward(self, x):
        identity = x

        if self.stride == 2:
            out = F.avg_pool2d(x, kernel_size=(2, 2))
        else:
            out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.1, training=self.training)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class _ResNet(nn.Module):
    def __init__(self, layers, groups=1, width_per_group=64, norm_layer=None):
        super(_ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        self.groups = groups
        self.base_width = width_per_group

        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

    #this should be resnetbasicblock
    def _make_layer(self, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes:
            if stride == 1:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=1, bias=False),
                    norm_layer(planes),
                )
                # init_layer(downsample[0])
                # init_bn(downsample[1])
            elif stride == 2:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2),
                    nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=1, bias=False),
                    norm_layer(planes),
                )
                # init_layer(downsample[1])
                # init_bn(downsample[2])

        layers = []
        layers.append(_ResnetBasicBlock(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(_ResnetBasicBlock(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

class ResNet38(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
        fmax, classes_num):

        super(ResNet38, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
            win_length=window_size, window=window, center=center, pad_mode=pad_mode,
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db,
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
            freq_drop_width=8, freq_stripes_num=2)


        self.resnet = _ResNet(layers=[3, 4, 6, 3])



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
