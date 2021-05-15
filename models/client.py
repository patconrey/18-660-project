"""
This code was modified heavily from code found here: https://github.com/katsura-jp/fedavg.pytorch.
We use their basic infrastructure of a Client class that trains its own model. We decided that
this is the most realistic representation of how the federated learning schemes would be implemented
in the wild. However, we have to modify their code quite a bit in order to handle the variety of
experiments we perform.
"""

import random
from collections import OrderedDict
import numpy as np
import torch
import copy

class Client:
    def __init__(self,
                 client_id,
                 dataloader,
                 should_use_heterogeneous_E=False,
                 local_epoch=5,
                 local_epoch_min=1,
                 local_epoch_max=5,
                 device='cpu'):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device
        self.__model = None
        self.should_use_heterogeneous_E = should_use_heterogeneous_E
        self.local_epoch = local_epoch
        self.epochs_to_perform = local_epoch # This property will change per communication round if we're using heterogeneous E
        self.local_epoch_max = local_epoch_max
        self.local_epoch_min = local_epoch_min

    @property
    def model(self):
        return self.__model

    @property
    def tau(self):
        return np.floor(self.epochs_to_perform * len(self.dataloader.dataset) / self.dataloader.batch_size)

    @model.setter
    def model(self, model):
        self.__model = model

    def client_update(self, optimizer, optimizer_args, loss_fn, communication_round=0):
        raise NotImplementedError

    def __len__(self):
        return len(self.dataloader.dataset)


class FederatedClient(Client):
    def client_update(self, optimizer, optimizer_args, loss_fn, communication_round=0):
        self.model.train()
        self.model.to(self.device)
        
        # Decay learning rate.
        learning_rate = optimizer_args.lr
        if communication_round >= 100:
            learning_rate /= 2
        if communication_round >= 150:
            learning_rate /= 2

        optimizer = optimizer(self.model.parameters(), learning_rate)
        state_before_training = copy.deepcopy(self.model.state_dict())
        # Decide local epochs to use.
        epochs_to_perform = self.local_epoch
        if self.should_use_heterogeneous_E:
            epochs_to_perform = random.randint(self.local_epoch_min, self.local_epoch_max)
        self.epochs_to_perform = epochs_to_perform
        # only used for fednova
        # initialize list of gradients stored end of each batch round
        rounds_performed = int(epochs_to_perform*np.ceil(len(self.dataloader.dataset)/self.dataloader.batch_size))
        aggregation_weights = np.ones(rounds_performed)
        L1_aggregation = np.sum(aggregation_weights)
        grad_accumulator = OrderedDict()
        for n, p in self.model.named_parameters():
            if(p.requires_grad):
                grad_accumulator[n] = 0

        round_counter = 0
        for i in range(epochs_to_perform):
            for img, target in self.dataloader:
                optimizer.zero_grad()

                img = img.to(self.device)
                target = target.to(self.device)

                logits = self.model(img)
                loss = loss_fn(logits, target)

                pred = logits.argmax(dim=1, keepdim=True)
                
                loss.backward()
                optimizer.step()
                # gather and store gradients
                local_state = self.model.state_dict()
                for n, p in self.model.named_parameters():
                    if(p.requires_grad):
                        grad_accumulator[n] += p.grad * aggregation_weights[round_counter]/L1_aggregation
                round_counter += 1

        state_adjustment = OrderedDict()
        for key in self.model.state_dict().keys():
            state_adjustment[key] = self.model.state_dict()[key] - state_before_training[key]
        
        self.state_adjustment = state_adjustment
        self.prev_net_gradient = grad_accumulator

    def eval_train(self, loss_fn):
        # evaluate loss and accurary after training finished (for tracking training loss/accuracy)
        self.model.eval()
        self.model.to(self.device)
        loss = 0
        num_correct = 0
        with torch.no_grad():
            for img, target in self.dataloader:
                img = img.to(self.device)
                target = target.to(self.device)
                logits = self.model(img)
                loss += loss_fn(logits, target).item()
                pred = logits.argmax(dim=1, keepdim=True)
                num_correct += pred.eq(target.view_as(pred)).sum().item()
        
        train_loss = loss/len(self.dataloader.dataset)
        train_accuracy = (num_correct/len(self.dataloader.dataset))*100

        return train_loss, train_accuracy
