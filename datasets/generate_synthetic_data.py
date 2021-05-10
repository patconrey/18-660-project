import numpy as np
from numpy.random import RandomState, SeedSequence
import torch

# Generate synthetic data for federated learning using method
# described in:
#   Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, 
#   Ameet Talwalkar, and Virginia Smith. "Federated optimization
#   in heterogeneous networks". In Conference on Machine Learning 
#   and Systems, 2020. https://arxiv.org/pdf/1812.06127.pdf
# Approach to create synthetic data discribed in section 5.1

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)

def calc_labels(W,X):
    Z = torch.mm(torch.Tensor(W),torch.Tensor(X))
    Zexp = torch.exp(Z)
    Zexp_sums = torch.transpose(torch.sum(Zexp, axis=0).unsqueeze(1), 0, 1)
    labels = torch.argmax((Zexp / Zexp_sums),dim=0)
    labels_onehot = torch.nn.functional.one_hot(labels, num_classes=W.shape[0])
    # labels_onehot = np.zeros((Z.shape[0], Z.shape[1]))
    # labels_onehot[labels, np.arange(Z.shape[1])] = 1
    return labels_onehot

class SyntheticDataGenerator(object):
    def __init__(self, alpha=1, beta=1, n_features=60, n_classes=10):
        # "alpha controls how much local models differ from each other"
        self.alpha = alpha
        # "beta controls how much the local data at each device differs 
        # from that of other devices"
        self.beta = beta
        # self.rng = np.random.default_rng(seed=1234567)
        self.rng = np.random.default_rng()
        self.n_features = n_features
        self.n_classes = n_classes
        self.feature_covariance_matrix = np.diag(np.array([i**(-1.2) for i in np.arange(1,n_features+1)]))
    
    def generate_client_data(self, n_samples, client_id):
        B = self.rng.normal(0, self.beta)
        v = self.rng.normal(B, 1, (self.n_features,))
        X = self.rng.multivariate_normal(v, self.feature_covariance_matrix, size=n_samples).transpose()
        bias_row = np.ones((1,n_samples))
        X = np.concatenate((bias_row, X), axis=0)
        # size X should be n_features * n_samples
        u = self.rng.normal(0, self.alpha)
        W = self.rng.normal(u, 1, (self.n_classes, self.n_features))
        b = self.rng.normal(u, 1, (self.n_classes,1))
        W = np.concatenate((b,W),axis=1)
        labels = calc_labels(W,X)
        dataset = SyntheticLocalDataset(X, labels, W, client_id)
        return dataset

    def generate_iid_client_data(self, client_samples):
        u = self.rng.normal(0, self.alpha)
        W = self.rng.normal(u, 1, (self.n_classes, self.n_features))
        b = self.rng.normal(u, 1, (self.n_classes,1))
        W = np.concatenate((b,W),axis=1)
        v = np.zeros((self.n_features,))
        iid_data = []
        for client_id in np.arange(len(client_samples)):
            n_samples = client_samples[client_id]
            X = self.rng.multivariate_normal(v, self.feature_covariance_matrix, size=n_samples).transpose()
            bias_row = np.ones((1,n_samples))
            X = np.concatenate((bias_row, X), axis=0)
            # size X should be n_features * n_samples
            # labels = np.argmax(softmax((W@X)), axis=0)
            # labels_onehot = np.zeros((self.n_classes, n_samples))
            # labels_onehot[labels, np.arange(n_samples)] = 1
            labels = calc_labels(W,X)
            dataset = SyntheticLocalDataset(X, labels, W, client_id)
            iid_data.append(dataset)
        return iid_data

class SyntheticLocalDataset(object):
    def __init__(self, features, labels, W, client_id):
        self.features = features
        self.labels = labels
        self.W_true = W
        self.client_id = client_id

    def __getitem__(self, index):
        feature_set = self.features[:,index]
        label = self.labels[index]
        return feature_set, label

    def __len__(self):
        return len(self.labels)

def create_synthetic_lr_datasets(num_clients=30,
                                 alpha=1,
                                 beta=1,
                                 iid=False):
    # This is my best attempt so far at generating the number of samples for
    # each client in a distribution that "follows a power law", as the paper
    # states. We can continue to tweak this.
    zipf_param = 2
    client_samples = np.random.zipf(zipf_param, size=(100*num_clients,))
    client_samples = np.sort(client_samples)[-num_clients:]
    # from matplotlib import pyplot as plt
    # plt.hist(client_samples, bins=np.arange(min(client_samples), max(client_samples)+1,))
    # plt.show()
    # half the number of test samples as total training? more? less?
    n_test_samples = int(client_samples.sum()/2)
    data_generator = SyntheticDataGenerator(alpha, beta)
    
    if iid:
        client_samples = np.append(client_samples, n_test_samples)
        local_datasets = data_generator.generate_iid_client_data(client_samples)
        # not sure how to make the test dataset?
        test_dataset = local_datasets[-1]
        local_datasets = local_datasets[:-1]
        test_dataset.client_id = -1
    else:
        local_datasets = []
        for client_id in range(num_clients):
            n_samples = client_samples[client_id]
            client_data = data_generator.generate_client_data(n_samples, client_id)
            local_datasets.append(client_data)
        test_dataset = data_generator.generate_client_data(n_test_samples, -1)
    
    return (local_datasets, test_dataset)
        