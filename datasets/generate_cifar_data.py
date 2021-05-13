"""
Much of dataset-generation code was taken from https://github.com/IBM/FedMA/.
The linked project is a citaiton of the FedNova paper, which also borrows
its dataset-generation scheme.
"""

import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10 as CifarDataset
from torchvision.transforms import transforms


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

class CIFAR10(Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        cifar_dataobj = CifarDataset(self.root, self.train, self.transform, self.target_transform, self.download)

        if self.train:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def load_cifar10_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)


# define a function to get the number of samples at each client
def get_dirichlet_samples_for_client(y_train, num_clients):
    min_size = 0
    K = 10
    N = y_train.shape[0]
    net_dataidx_map = {}

    while min_size < 10:
        idx_batch = [[] for _ in range(num_clients)]
        # for each class in the dataset
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(0.1, num_clients))
            ## Balance
            proportions = np.array([p*(len(idx_j)<N/num_clients) for p,idx_j in zip(proportions,idx_batch)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(num_clients):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    return 


def partition_data(data_root, num_clients, should_use_heterogeneous_data=True, iid=False):
    X_train, y_train, _, _ = load_cifar10_data(data_root)
    n_train = X_train.shape[0]

    if not should_use_heterogeneous_data:
        if iid:
            idxs = np.random.permutation(n_train)
            batch_idxs = np.array_split(idxs, num_clients)
            net_dataidx_map = {i: batch_idxs[i] for i in range(num_clients)}

            return net_dataidx_map

        elif not iid:
            sorted_indices = np.argsort(y_train)
            batch_idxs = np.array_split(sorted_indices, num_clients)
            net_dataidx_map = {i: batch_idxs[i] for i in range(num_clients)}

            return net_dataidx_map

    elif should_use_heterogeneous_data:
        if iid:
            min_size = 0
            K = 10
            N = y_train.shape[0]
            net_dataidx_map = {}

            while min_size < 10:
                idx_batch = [[] for _ in range(num_clients)]
                # for each class in the dataset
                for k in range(K):
                    idx_k = np.where(y_train == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(0.1, num_clients))
                    ## Balance
                    proportions = np.array([p*(len(idx_j)<N/num_clients) for p,idx_j in zip(proportions,idx_batch)])
                    proportions = proportions/proportions.sum()
                    proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])

            for j in range(num_clients):
                np.random.shuffle(idx_batch[j])
                net_dataidx_map[j] = idx_batch[j]

            return net_dataidx_map

        elif not iid:
            raise NotImplementedError()


def create_datasets_from_mapping(data_root, mapping):
    transform = transforms.Compose([transforms.ToTensor()])

    local_datasets = []
    for k in mapping.keys():
        client_dataset = CIFAR10(data_root, dataidxs=mapping[k], train=True, download=False, transform=transform)
        local_datasets.append(client_dataset)
    
    test_dataset = CIFAR10(data_root, train=False, download=True, transform=transform)

    return (local_datasets, test_dataset)


def create_cifar_datasets(data_root, num_client, should_use_heterogeneous_data=True, iid=True):
    data_indices_for_clients = partition_data(data_root, num_client, should_use_heterogeneous_data=should_use_heterogeneous_data, iid=iid)
    local_datasets, test_set = create_datasets_from_mapping(data_root, data_indices_for_clients)

    return (local_datasets, test_set)

        
if __name__ == "__main__":
    data_indices_for_clients = partition_data('../data/', 30, should_use_heterogeneous_data=True, iid=True)
    local_datasets, test_set = create_datasets_from_mapping('../data/', data_indices_for_clients)

    print('hi')