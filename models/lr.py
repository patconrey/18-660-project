import torch
import torch.nn as nn

# class definition for multiclass logistic regression

class LR(nn.Module):
    def __init__(self, n_features=60, n_classes=10):
        super().__init__()

        self.n_features = n_features
        self.n_classes = n_features
        self.linear = nn.Linear(n_features, n_classes)
        # We don't need this because we're using CrossEntropyLoss as the objective function
        # self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        return self.linear(x.float())
        
