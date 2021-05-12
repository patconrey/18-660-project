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

 
class FedAvgCenterServer(CenterServer):
    def __init__(self, model, dataloader, device="cpu"):
        super().__init__(model, dataloader, device)

    def aggregation(self, clients, aggregation_weights, sample_set, scale_factor):
        # initialize update dictionary
        update_state = OrderedDict()
        for key in self.model.state_dict().keys():
            update_state[key] = 0

        for k, client in enumerate(clients):
            # only update using clients that were trained on in this round
            if client.client_id not in sample_set:
                continue
            local_state = client.model.state_dict()
            for key in self.model.state_dict().keys():
                update_state[key] += local_state[key] * aggregation_weights[k] * scale_factor

        self.model.load_state_dict(update_state)

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

        self.model.to("cpu")
        test_loss = test_loss / len(self.dataloader)
        accuracy = 100. * correct / len(self.dataloader.dataset)

        return test_loss, accuracy
    

class FedNovaCenterServer(CenterServer):
    def __init__(self, model, dataloader, device="cpu"):
        super().__init__(model, dataloader, device)

    def aggregation(self, clients, aggregation_weights, sample_set, scaling_factor):
        
        update_state = OrderedDict()
        
        for key in self.model.state_dict().keys():
            update_state[key] = 0
        
        trained_clients = []
        for ind in sample_set:
            trained_clients.append(clients[ind])

        taus = np.array([client.tau for client in trained_clients])
        pks = np.array([len(client.dataloader.dataset) for client in trained_clients])
        pks = pks / pks.sum()
        tau_eff = taus @ pks
        
        for k, client in enumerate(clients):
            # only update using clients that were trained on in this round
            if client.client_id not in sample_set:
                continue
            local_state = client.model.state_dict()
            for key in self.model.state_dict().keys():
                update_state[key] += local_state[key] * aggregation_weights[k] / client.tau * scaling_factor

        for key in self.model.state_dict().keys():
            update_state[key] *= tau_eff

        self.model.load_state_dict(update_state)

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

        self.model.to("cpu")
        test_loss = test_loss / len(self.dataloader)
        accuracy = 100. * correct / len(self.dataloader.dataset)

        return test_loss, accuracy
