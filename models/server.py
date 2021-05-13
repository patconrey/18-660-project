"""
This code was modified heavily from code found here: https://github.com/katsura-jp/fedavg.pytorch.
We use their basic infrastructure of a Server class that handles aggregating the weights of the
clients. However, we have to modify their code quite a bit in order to handle the variety of
experiments we perform.
"""

import numpy as np
import copy
from collections import OrderedDict
import torch

class CenterServer:
    def __init__(self, model, dataloader, device="cpu"):
        self.model = model
        self.dataloader = dataloader
        self.device = device

    def aggregation(self):
        raise NotImplementedError

    def send_model(self):
        return copy.deepcopy(self.model)

    def validation(self):
        raise NotImplementedError

    def eval_all_train_data(self, clients, loss_fn, aggregation_weights):
        num_clients = len(clients)
        losses = np.zeros(num_clients)
        accuracies = np.zeros(num_clients)
        for ind, client in enumerate(clients):
            loss, accuracy = client.eval_train(loss_fn)
            losses[ind] = loss
            accuracies[ind] = accuracy

        train_loss = np.dot(aggregation_weights, losses)
        train_accuracy = np.dot(aggregation_weights, accuracies)
        return train_loss, train_accuracy


class FedAvgCenterServer(CenterServer):
    def __init__(self, model, dataloader, device="cpu"):
        super().__init__(model, dataloader, device)

    def aggregation(self, trained_clients, trained_clients_aggregation_weights, scaling_factor):
        # NOTE: we only update using clients that were trained on in this round
        global_model_state = copy.deepcopy(self.model.state_dict())
        # initialize state dictionary for compiling the updates to the model
        update_state = OrderedDict()
        for key in self.model.state_dict().keys():
            update_state[key] = 0
            
        for k, client in enumerate(trained_clients):
            for key in self.model.state_dict().keys():
                update_state[key] += client.state_adjustment[key] * trained_clients_aggregation_weights[k] * scaling_factor

        for key in self.model.state_dict().keys():
            global_model_state[key] += update_state[key]

        self.model.load_state_dict(global_model_state)

    def validation(self, loss_fn):
        self.model.to(self.device)
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for features, label in self.dataloader:
                features = features.to(self.device)
                label = label.to(self.device)
                logits = self.model(features)
                test_loss += loss_fn(logits, label).item()
                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()

        self.model.to(self.device)
        test_loss = test_loss / len(self.dataloader)
        accuracy = 100. * correct / len(self.dataloader.dataset)

        return test_loss, accuracy
    

class FedNovaCenterServer(CenterServer):
    def __init__(self, model, dataloader, device="cpu"):
        super().__init__(model, dataloader, device)

    def aggregation(self, trained_clients, trained_clients_aggregation_weights, scaling_factor):
        # NOTE: we only update using clients that were trained on in this round
        global_model_state = copy.deepcopy(self.model.state_dict())
        # initialize state dictionary for compiling the updates to the model
        update_state = OrderedDict()
        for key in self.model.state_dict().keys():
            update_state[key] = 0

        # calculating tau_eff is key for FedNova
        taus = np.array([client.tau for client in trained_clients])
        pks = np.array([len(client.dataloader.dataset) for client in trained_clients])
        pks = pks / pks.sum()
        tau_eff = taus @ pks

        for k, client in enumerate(trained_clients):
            for key in self.model.state_dict().keys():
                update_state[key] += client.state_adjustment[key] * trained_clients_aggregation_weights[k] / client.tau * scaling_factor

        for key in self.model.state_dict().keys():
            global_model_state[key] += (tau_eff*update_state[key])

        self.model.load_state_dict(global_model_state)

    def validation(self, loss_fn):
        self.model.to(self.device)
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for img, target in self.dataloader:
                img = img.to(self.device)
                target = target.to(self.device)
                logits = self.model(img)
                test_loss += loss_fn(logits, target).item()
                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        self.model.to(self.device)
        test_loss = test_loss / len(self.dataloader)
        accuracy = 100. * correct / len(self.dataloader.dataset)

        return test_loss, accuracy
