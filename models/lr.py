import torch
import torch.nn as nn
from torch.nn.functional import log_softmax

# class definition for multiclass logistic regression

class LR(nn.Module):
    def __init__(self, in_features=60, num_classes=10):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return log_softmax(self.linear(x.float()), dim=1)
