import numpy as np
from numpy.random import RandomState, SeedSequence
import torch
from matplotlib import pyplot as plt

# Generate synthetic data for federated learning using method
# described in:
#   Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, 
#   Ameet Talwalkar, and Virginia Smith. "Federated optimization
#   in heterogeneous networks". In Conference on Machine Learning 
#   and Systems, 2020. https://arxiv.org/pdf/1812.06127.pdf
# Approach to create synthetic data discribed in section 5.1
# and Appendix C

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)

def calc_labels(W, X, b):
    z = np.dot(W, X) + b
    a = softmax(z)
    y = a.argmax(axis=0)
    return y

def generate_client_samples_lognormal(num_clients, min_samples, mean_samples, std_samples, show_cdf=False):
    client_samples = min_samples + np.random.lognormal(mean_samples, std_samples, num_clients).astype(int)
    if show_cdf:
        pct_of_total = np.flip(np.cumsum(np.flip(np.sort(client_samples)))/np.sum(client_samples))
        plt.plot(np.arange(len(client_samples)), pct_of_total)
        plt.yscale('log')
        plt.xscale('log')
        plt.show()
    return client_samples

def generate_client_samples_zipf(num_clients, zipf_param, show_cdf=False):
    client_samples = np.random.zipf(zipf_param, size=(100*num_clients,))
    client_samples = np.sort(client_samples)[-num_clients:]
    if show_cdf:
        pct_of_total = np.flip(np.cumsum(np.flip(client_samples_sorted))/np.sum(client_samples))
        plt.plot(np.arange(len(client_samples)), pct_of_total)
        plt.yscale('log')
        plt.xscale('log')
        plt.show()
    return client_samples

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
        self.feature_covariance_matrix = np.diag(np.power(np.arange(1,n_features+1), -1.2))
    
    def generate_client_data(self, client_samples, iid):
        n_clients = len(client_samples)
        feat_set_list = []
        label_set_list = []
        if iid:
            u_iid = self.rng.normal(0, self.alpha)
            W_iid = self.rng.normal(u_iid, 1, (self.n_classes, self.n_features))
            v_iid = np.zeros((self.n_features,))
            b_iid = self.rng.normal(u_iid, 1, (self.n_classes,1))
        for client_id in np.arange(n_clients):
            n_samples = client_samples[client_id]
            if iid:
                W = W_iid
                b = b_iid
                v = v_iid
            else:
                u = self.rng.normal(0, self.alpha)
                W = self.rng.normal(u, 1, (self.n_classes, self.n_features))
                b = self.rng.normal(u, 1, (self.n_classes, 1))
                B = self.rng.normal(0, self.beta)
                v = self.rng.normal(B, 1, (self.n_features,))
            X = self.rng.multivariate_normal(v, self.feature_covariance_matrix, size=n_samples).transpose()
            labels = calc_labels(W, X, b)
            feat_set_list.append(X)
            label_set_list.append(labels)
        return feat_set_list, label_set_list

class SyntheticLocalDataset(object):
    def __init__(self, features, labels, client_id):
        self.features = features
        self.labels = labels
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
                                 n_features=60,
                                 n_classes=10,
                                 should_use_heterogeneous_data=True,
                                 iid=False):
    if should_use_heterogeneous_data:
        client_samples = generate_client_samples_lognormal(num_clients, 100, 4, 1)
        print(client_samples)
    else:
        client_samples = np.ones(num_clients).astype(int)*120
    # client_samples = generate_client_samples_zipf(num_clients, 100, 2)
    # half the number of test samples as total training? more? less?
    data_generator = SyntheticDataGenerator(alpha, beta, n_features, n_classes)
    
    feat_set_list, label_set_list = data_generator.generate_client_data(client_samples, iid)
    pct_train = 0.9
    force_equal_test_distribution = True
    equal_split_index = int((1-pct_train)*min(client_samples))
    test_features_set = np.empty((n_features,0), dtype=float)
    test_labels_set = np.empty((1,0), dtype=int)
    local_datasets = []
    for client_id in np.arange(num_clients):
        features = feat_set_list[client_id]
        labels = label_set_list[client_id]
        if force_equal_test_distribution:
            split_index = equal_split_index
        else:
            split_index = int(pct_train*features.shape[1])
        train_features = features[:,:split_index]
        train_labels = labels[:split_index]
        test_features = features[:,split_index:]
        test_labels = labels[split_index:]
        local_datasets.append(SyntheticLocalDataset(train_features, train_labels, client_id))
        test_features_set = np.concatenate((test_features_set, test_features), axis=1)
        test_labels_set = np.append(test_labels_set, test_labels)

    test_dataset = SyntheticLocalDataset(test_features_set, test_labels_set, -1)

    # show distribution of classes across clients
    if False:
        class_bins=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])-.5
        f = plt.figure(figsize=(14, 9))
        for i in range(30):
            plt.subplot(6, 5, i+1)
            plt.title('Client {}'.format(i))
            plt.hist(
                    local_datasets[i].labels,
                    density=False,
                    bins=class_bins)
        plt.show()
    
    return (local_datasets, test_dataset)
        