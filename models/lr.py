import torch
import torch.nn as nn

# class definition for multiclass logistic regression

class LR(nn.Module):
    def __init__(self, in_features=60, num_classes=10, using_cross_entropy_loss=True):
        super().__init__()

        self.in_features = in_features
        self.num_classes = num_classes
        self.using_cross_entropy_loss = using_cross_entropy_loss
        self.linear = nn.Linear(in_features, num_classes)
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        if self.using_cross_entropy_loss:
            return self.linear(x.float())
        else:
            return self.activation(self.linear(x.float()))
