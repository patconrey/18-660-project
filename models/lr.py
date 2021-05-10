import torch
import torch.nn as nn

# class definition for multiclass logistic regression

class LR(nn.Module):
    def __init__(self, in_features=60, num_classes=10):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.linear = nn.Linear(in_features, num_classes)
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        # I've read that pytorch cross-entropy loss expects raw logits, even for LR
        #return self.linear(x.float())
        return self.activation(self.linear(x.float()))
