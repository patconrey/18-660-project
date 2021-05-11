import pandas as pd
import sklearn.datasets
from datasets.generate_synthetic_data import create_synthetic_lr_datasets
from models.lr import LR
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, NLLLoss
from torch.optim import *

class SyntheticLocalDataset(object):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __getitem__(self, index):
        feature_set = self.features[index,:]
        label = self.labels[index]
        return feature_set, label

    def __len__(self):
        return self.features.shape[0]

# synthetic data generation code taken from
# https://botbark.com/2019/12/28/creating-synthetic-data-for-logistic-regression/

use_easy_synthetic_data = False
if use_easy_synthetic_data:
    N = 4 # num features
    K = 3 # num classes
    data = sklearn.datasets.make_classification(n_samples=1500, n_classes=K, n_clusters_per_class=1, n_features=N, n_informative=N, n_redundant=0, n_repeated=0)
    x_tr = data[0][:1000,:]
    y_tr = data[1][:1000]
    x_test = data[0][1000:,:]
    y_test = data[1][1000:]

    # create dataloaders
    train_dataset = SyntheticLocalDataset(x_tr, y_tr)
    test_dataset = SyntheticLocalDataset(x_test, y_test)
else:
    N = 60 # num features
    K = 10 # num classes
    (train_datasets, test_dataset) = create_synthetic_lr_datasets(1, 1, 1, N, K, True)
    train_dataset = train_datasets[0]

dataloader = DataLoader(train_dataset,
                            num_workers=0,
                            batch_size=50,
                            shuffle=True)
test_dataloader = DataLoader(train_dataset,
                            num_workers=0,
                            batch_size=50,
                            shuffle=True)

use_cross_entropy_loss = True
if use_cross_entropy_loss:
    loss_fn = CrossEntropyLoss()
else:
    loss_fn = NLLLoss()

model = LR(N, K, use_cross_entropy_loss)

model.train()
model.to('cpu')
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)

for i in range(100):

    for feat, label in dataloader:
        feat = feat.to('cpu')
        label = label.to('cpu')
        optimizer.zero_grad()
        logits = model(feat)
        loss = loss_fn(logits, label)
        loss.backward()
        optimizer.step()

    model.eval()
    test_loss = 0
    correct = 0
    for feat, label in test_dataloader:
        feat = feat.to('cpu')
        label = label.to('cpu')
        logits = model(feat)
        test_loss += loss_fn(logits, label).item()
        pred = logits.argmax(dim=1, keepdim=True)
        correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss = test_loss / len(test_dataloader)
    accuracy = 100. * correct / len(test_dataloader.dataset)
    print("Epoch %d: test loss %f, test accuracy %f pct" % (i, test_loss, accuracy))
    model.train()
    
    model.to("cpu")