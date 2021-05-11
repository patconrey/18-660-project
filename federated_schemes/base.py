import random

import torch
import numpy as np

from datasets.mnist import MnistLocalDataset
from datasets.generate_synthetic_data import create_synthetic_lr_datasets
from utils.data import get_mnist_data


class FedBase:
    def create_mnist_datasets(self,
                              num_clients=100,
                              shard_size=300,
                              datadir="./data/mnist",
                              iid=False):
        train_img, train_label, test_img, test_label = get_mnist_data(datadir)

        train_sorted_index = np.argsort(train_label)
        train_img = train_img[train_sorted_index]
        train_label = train_label[train_sorted_index]

        if iid:
            random.shuffle(train_sorted_index)
            train_img = train_img[train_sorted_index]
            train_label = train_label[train_sorted_index]

        shard_start_index = [i for i in range(0, len(train_img), shard_size)]
        random.shuffle(shard_start_index)
        print(
            f"divide data into {len(shard_start_index)} shards of size {shard_size}"
        )

        num_shards = len(shard_start_index) // num_clients
        local_datasets = []
        for client_id in range(num_clients):
            _index = num_shards * client_id
            img = np.concatenate([
                train_img[shard_start_index[_index +
                                            i]:shard_start_index[_index + i] +
                          shard_size] for i in range(num_shards)
            ],
                                 axis=0)

            label = np.concatenate([
                train_label[shard_start_index[_index +
                                              i]:shard_start_index[_index +
                                                                   i] +
                            shard_size] for i in range(num_shards)
            ],
                                   axis=0)

            local_datasets.append(MnistLocalDataset(img, label, client_id))

        test_sorted_index = np.argsort(test_label)
        test_img = test_img[test_sorted_index]
        test_label = test_label[test_sorted_index]

        test_dataset = MnistLocalDataset(test_img, test_label, client_id=-1)

        return local_datasets, test_dataset

    def create_synthetic_lr_datasets(self,
                              num_clients=30,
                              alpha=1,
                              beta=1,
                              should_use_heterogeneous_data=False,
                              iid=False):
        # This is my best attempt so far at generating the number of samples for
        # each client in a distribution that "follows a power law", as the paper
        # states. We can continue to tweak this.
        zipf_param = 2
        client_samples = np.random.zipf(zipf_param, size=(10*num_clients,))
        client_samples = np.sort(client_samples)[-num_clients:]
        # half the number of test samples as total training? more? less?
        n_test_samples = int(client_samples.sum()/2)
        data_generator = SyntheticDataGenerator(alpha, beta)
        
        if iid:
            local_datasets = data_generator.generate_iid_client_data(client_samples)
            # not sure how to make the test dataset?
            test_dataset = data_generator.generate_iid_client_data([n_test_samples])
            test_dataset = test_dataset[0]
        else:
            local_datasets = []
            for client_id in range(num_clients):
                local_datasets.append(data_generator.generate_client_data(n_samples, client_id))
            test_dataset
        test_dataset['client_id'] = -1

        return local_datasets, test_dataset
    
    def train_step(self):
        raise NotImplementedError

    def validation_step(self):
        raise NotImplementedError

    def fit(self, num_round):
        raise NotImplementedError
