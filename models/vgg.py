"""
This code is a re-implementation of a standard VGG model with modifications
to handle the smaller-than-expected size of CIFAR data. It is based off of
code found here: https://github.com/JYWa/FedNova.
"""

import numpy as np
import torch.nn as nn

config = {}
config['A'] = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
config['B'] = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
config['D'] = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
config['E'] = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
               512, 512, 512, 512, 'M']

class VGG_32x32(nn.Module):
    def __init__(self, version='A', batch_normalization=False):
        '''
        VGG-11 compatible with 32x32 input images from CIFAR-10
        :param version: Which version of VGG-11 to use (see https://arxiv.org/pdf/1409.1556.pdf)
        :param batch_normalization: Whether to use batch normalization
        '''
        super(VGG_32x32, self).__init__()

        # Create layers for chosen flavor of VGG
        layers = []
        in_channels = 3
        relu = nn.ReLU(inplace=True)
        for elem in config[version][0]:
            if elem == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                out_channels = elem
                conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
                if batch_normalization:
                    batch_norm = nn.BatchNorm2d(num_features=out_channels)
                    layers += [conv2d, batch_norm, relu]
                else:
                    layers += [conv2d, relu]
                in_channels = elem
        self.sequence = nn.Sequential(*layers)

        # Classifier to output 10 class labels
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )

        # Initialize weights to be zero mean, sqrt(2/n) variance
        for mod in self.modules():
            if type(mod) == nn.Conv2d:
                n = np.prod(mod.kernel_size) * mod.out_channels
                mod.weight.data.normal_(0, np.sqrt(2/n))
                mod.bias.data.zero_()

    # Method for forward propagation
    def forward(self, x):
        y1 = self.sequence(x)
        y2 = y1.view(y1.size(0), -1)
        y3 = self.classifier(y2)
        return y3

# VGG compatible with 32x32 input images (no batch norm)
def vgg11():
    model = VGG_32x32(version='A', batch_normalization=False)
    return model

# VGG compatible with 32x32 input images (batch norm)
def vgg11_bn():
    model = VGG_32x32(version='A', batch_normalization=True)
    return model
