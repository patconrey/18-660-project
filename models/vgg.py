import torch.nn as nn
from torchvision import models


class VGG(nn.Module):
    def __init__(self, in_features, num_classes, hidden_dim, use_pretrained=True):
        '''
        :param in_features: Unused
        :param num_classes: Number of classes
        :param hidden_dim: Unused
        :param use_pretrained: Whether to load pretrained weights
        '''
        super(VGG, self).__init__()
        self.model = models.vgg11(pretrained=use_pretrained)
        in_features_classifier = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features=in_features_classifier, out_features=num_classes, bias=True)

    def forward(self, x):
        return self.model.forward(x)
