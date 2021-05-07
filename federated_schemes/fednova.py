import logging
log = logging.getLogger(__name__)

import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from models.client import FedAvgClient as Client
from models.server import FedNovaCenterServer as CenterServer

from datasets.mnist import MnistLocalDataset
from datasets.generate_synthetic_data import create_synthetic_lr_datasets
from utils.data import get_mnist_data


class FedNova():
    def __init__(self,
                 model,
                 optimizer,
                 optimizer_args,
                 num_clients=200,
                 batchsize=50,
                 fraction=1,
                 iid=False,
                 should_use_heterogeneous_data=False,
                 should_use_heterogeneous_E=False,
                 local_epoch=1,
                 local_epoch_min=1,
                 local_epoch_max=5,
                 device="cpu",
                 writer=None):
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args

        self.num_clients = num_clients  # K
        self.batchsize = batchsize  # B
        self.fraction = fraction  # C, 0 < C <= 1
        self.local_epoch = local_epoch  # E

        local_datasets, test_dataset = self.create_mnist_datasets(
            num_clients,
            iid=iid,
            should_use_heterogeneous_data=should_use_heterogeneous_data)
            
        local_dataloaders = [
            DataLoader(dataset,
                       num_workers=0,
                       batch_size=batchsize,
                       shuffle=True) for dataset in local_datasets
        ]

        self.clients = [
            Client(k,
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
        self.center_server = CenterServer(model, test_dataloader, device)

        self.loss_fn = CrossEntropyLoss()

        self.writer = writer

        self._round = 0
        self.result = None

    def fit(self, num_round):
        self._round = 0
        self.result = {'loss': [], 'accuracy': []}
        self.validation_step()
        for t in range(num_round):
            self._round = t + 1
            self.train_step()
            self.validation_step()

    def train_step(self):
        self.send_model()
        n_sample = max(int(self.fraction * self.num_clients), 1)
        sample_set = np.random.randint(0, self.num_clients, n_sample)
        for k in iter(sample_set):
            self.clients[k].client_update(
                self.optimizer,
                self.optimizer_args,
                self.loss_fn)
        self.center_server.aggregation(self.clients, self.aggregation_weights)

    def send_model(self):
        for client in self.clients:
            client.model = self.center_server.send_model()

    def validation_step(self):
        test_loss, accuracy = self.center_server.validation(self.loss_fn)
        log.info(
            f"[Round: {self._round: 04}] Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%"
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
