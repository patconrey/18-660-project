import logging

log = logging.getLogger(__name__)

import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from models.client import *
from models.server import *

from datasets.mnist import create_mnist_datasets
from datasets.generate_synthetic_data import create_synthetic_lr_datasets

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
            local_datasets, test_dataset = create_mnist_datasets(
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
        self.aggregation_weights = np.array([
            len(client) / self.total_data_size for client in self.clients
        ])

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
        self.send_model()
        self.validation_step()
        for t in range(num_round):
            self._round = t + 1
            self.train_step()
            self.send_model()
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
        # scaling_factor = self.num_clients/n_sample
        # In reality, I think this messes stuff up and they included this term only for their convergence
        # analysis, so just set it to 1 (equivalent to ignoring it)
        scaling_factor = 1

        trained_clients = []
        trained_clients_aggregation_weights = []

        num_samples_train = np.sum([len(self.clients[k]) for k in sample_set])
        for k in iter(sample_set):
            self.clients[k].client_update(
                self.optimizer,
                self.optimizer_args,
                self.loss_fn,
                communication_round=self._round)
            trained_clients.append(self.clients[k])
            trained_clients_aggregation_weights.append(self.aggregation_weights[k])

        self.center_server.aggregation(trained_clients, trained_clients_aggregation_weights, scaling_factor)

    def send_model(self):
        for client in self.clients:
            client.model = self.center_server.send_model()

    def validation_step(self):
        train_loss, train_accuracy = self.center_server.eval_all_train_data(self.clients, self.loss_fn, self.aggregation_weights)
        test_loss, test_accuracy = self.center_server.validation(self.loss_fn)

        log.info(
            f"[Round: {self._round: 04}] Train set: Average loss: {train_loss:.4f} Accuracy: {train_accuracy:.2f} // Test set: Average loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%"
        )
        if self.writer is not None:
            self.writer.add_scalar("train/loss", train_loss, self._round)
            self.writer.add_scalar("train/accuracy", train_accuracy, self._round)
            self.writer.add_scalar("val/loss", test_loss, self._round)
            self.writer.add_scalar("val/accuracy", test_accuracy, self._round)

        self.result['train_loss'].append(test_loss)
        self.result['train_accuracy'].append(train_accuracy)
        self.result['loss'].append(test_loss)
        self.result['accuracy'].append(test_accuracy)


    