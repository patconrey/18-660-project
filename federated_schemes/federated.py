import logging

log = logging.getLogger(__name__)

import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from models.client import *
from models.server import *

from datasets.mnist import MnistLocalDataset
from datasets.generate_synthetic_data import create_synthetic_lr_datasets
from utils.data import get_mnist_data

import matplotlib.pyplot as plt


class FederatedScheme():
    def __init__(self,
                 model,
                 optimizer,
                 optimizer_args,
                 num_clients=200,
                 batchsize=50,
                 fraction=1,
                 iid=False,
                 dataset='mnist',
                 should_use_heterogeneous_data=False,
                 should_use_heterogeneous_E=False,
                 local_epoch=1,
                 local_epoch_min=1,
                 local_epoch_max=5,
                 federated_type='fedavg',
                 device="cpu",
                 writer=None):

        assert federated_type in ['fedavg', 'fednova'], 'Unsupported federated_type {}'.format(federated_type)
        self.federated_type = federated_type

        self.optimizer = optimizer
        self.optimizer_args = optimizer_args

        self.num_clients = num_clients  # K
        self.batchsize = batchsize  # B
        self.fraction = fraction  # C, 0 < C <= 1
        self.local_epoch = local_epoch  # E
        self.dataset = dataset
        if dataset == 'mnist':
            local_datasets, test_dataset = self.create_mnist_datasets(
                num_clients,
                iid=iid,
                should_use_heterogeneous_data=should_use_heterogeneous_data)
        elif dataset == 'synthetic':
            (local_datasets, test_dataset) = create_synthetic_lr_datasets(num_clients,
                    alpha=1,
                    beta=1,
                    n_features=60,
                    n_classes=10,
                    should_use_heterogeneous_data=should_use_heterogeneous_data,
                    iid=iid)
        else:
            raise Exception("Unrecognized dataset argument")
        
        local_dataloaders = [DataLoader(dataset,
                            num_workers=0,
                            batch_size=batchsize,
                            shuffle=True) for dataset in local_datasets
                            ]

        self.clients = [
            FederatedClient(k,
                local_dataloaders[k],
                should_use_heterogeneous_E=should_use_heterogeneous_E,
                local_epoch=local_epoch,
                local_epoch_min=local_epoch_min,
                local_epoch_max=local_epoch_max,
                device=device) for k in range(num_clients)
        ]
        self.total_data_size = sum([len(client) for client in self.clients])
        self.aggregation_weights = [
            len(client) / self.total_data_size for client in self.clients
        ]

        test_dataloader = DataLoader(test_dataset,
                                     num_workers=0,
                                     batch_size=batchsize)
                                    
        self.center_server = None
        if self.federated_type == 'fedavg':
            self.center_server = FedAvgCenterServer(model, test_dataloader, device)
        elif self.federated_type == 'fednova':
            self.center_server = FedNovaCenterServer(model, test_dataloader, device)

        print('Server:', self.center_server)
        self.loss_fn = CrossEntropyLoss()

        self.writer = writer

        self._round = 0
        self.result = None

    def fit(self, num_round):
        self._round = 0
        self.result = {'loss': [], 'accuracy': [], 'train_loss': [], 'train_accuracy': []}
        self.validation_step()
        for t in range(num_round):
            self._round = t + 1
            self.train_step()
            self.validation_step()

        fig, axs = plt.subplots(2, 2)
        axs[0][0].plot(self.result['loss'])
        axs[0][0].set_title('Validation Loss')
        axs[0][1].plot(self.result['accuracy'])
        axs[0][1].set_title('Validation Accuracy')
        axs[1][0].plot(self.result['train_loss'])
        axs[1][0].set_title('Training Loss')
        axs[1][1].plot(self.result['train_accuracy'])
        axs[1][1].set_title('Training Accuracy')
        plt.show()

    def train_step(self):
        self.send_model()
       
        n_sample = max(int(self.fraction * self.num_clients), 1)
        # we can choose to sample with our without replacement - randint replaces, choice does not
        sample_set = np.random.choice(np.arange(self.num_clients), size=n_sample, replace=False)
        
        # Scaling factor is based on Footnote 1 of Wang et al:
        # "weighted averaging local changes, where the weight of client 
        #  i is re-scaled to (p_i m)/q." m is total samples, q is # clients samples, m is total # samples
        scaling_factor = self.num_clients/n_sample
        
        train_loss = 0
        train_correct = 0
        num_samples_train = np.sum([len(self.clients[k]) for k in sample_set])
        for k in iter(sample_set):
            self.clients[k].client_update(
                self.optimizer,
                self.optimizer_args,
                self.loss_fn,
                communication_round=self._round)
            train_loss += self.clients[k].most_recent_avg_loss*len(self.clients[k])
            train_correct += self.clients[k].most_recent_num_correct
        
        train_loss /= num_samples_train
        train_accuracy = train_correct/num_samples_train * 100
        if self.writer is not None:
            self.writer.add_scalar("test/loss", train_loss, self._round)
            self.writer.add_scalar("test/accuracy", train_accuracy, self._round)
        self.result['train_loss'].append(train_loss)
        self.result['train_accuracy'].append(train_accuracy)
        self.center_server.aggregation(self.clients, self.aggregation_weights, sample_set, scaling_factor)

    def send_model(self):
        for client in self.clients:
            client.model = self.center_server.send_model()

    def validation_step(self):
        test_loss, accuracy = self.center_server.validation(self.loss_fn)
        if self._round != 0:
            train_loss = self.result['train_loss'][-1]
            train_accuracy = self.result['train_accuracy'][-1]
        else:
            train_loss = test_loss
            train_accuracy = accuracy
        log.info(
            f"[Round: {self._round: 04}] Train set: Average loss: {train_loss:.4f} Accuracy: {train_accuracy:.2f} // Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%"
        )
        if self.writer is not None:
            self.writer.add_scalar("val/loss", test_loss, self._round)
            self.writer.add_scalar("val/accuracy", accuracy, self._round)

        self.result['loss'].append(test_loss)
        self.result['accuracy'].append(accuracy)

    def create_mnist_datasets(self,
                              num_clients=100,
                              datadir="./data/mnist",
                              should_use_heterogeneous_data=False,
                              iid=False):
        train_img, train_label, test_img, test_label = get_mnist_data(datadir)

        train_sorted_index = np.argsort(train_label)
        train_img = train_img[train_sorted_index]
        train_label = train_label[train_sorted_index]

        if iid:
            random.shuffle(train_sorted_index)
            train_img = train_img[train_sorted_index]
            train_label = train_label[train_sorted_index]

        # Set up hetereogenous / non-heterogeneous shard sizes
        shard_start_indices = None
        if should_use_heterogeneous_data:
            shard_sizes = np.random.power(3, size=num_clients)
            shard_sizes /= shard_sizes.sum()
            shard_sizes *= len(train_img)
            shard_sizes = shard_sizes.astype(int)
            shard_start_indices = np.hstack([[0], np.cumsum(shard_sizes)[:-1]])
        else:
            shard_size = int(train_img.shape[0] // num_clients)
            shard_start_indices = np.array([i for i in range(0, len(train_img), shard_size)])
            shard_sizes = np.array([shard_size for i in range(num_clients)])
        
        _indices_of_shards_and_sizes = np.arange(0, len(shard_start_indices))
        random.shuffle(_indices_of_shards_and_sizes)
        shuffled_shard_sizes = shard_sizes[_indices_of_shards_and_sizes]
        shuffled_shard_start_indices = shard_start_indices[_indices_of_shards_and_sizes]

        local_datasets = []
        for client_id in range(num_clients):
            start_index_of_client_data_shard = shuffled_shard_start_indices[client_id] 
            end_index_of_client_data_shard = start_index_of_client_data_shard + shuffled_shard_sizes[client_id]
            
            img = np.concatenate([
                train_img[start_index_of_client_data_shard : end_index_of_client_data_shard, :]
            ])
            label = np.concatenate([
                train_label[start_index_of_client_data_shard : end_index_of_client_data_shard]
            ])

            local_datasets.append(MnistLocalDataset(img, label, client_id))

        test_sorted_index = np.argsort(test_label)
        test_img = test_img[test_sorted_index]
        test_label = test_label[test_sorted_index]

        test_dataset = MnistLocalDataset(test_img, test_label, client_id=-1)

        return local_datasets, test_dataset

    