import random
import numpy as np

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
        self.most_recent_avg_loss = 0
        self.most_recent_num_correct = 0

    @property
    def model(self):
        return self.__model

    @property
    def tau(self):
        return np.floor( self.epochs_to_perform * len(self.dataloader.dataset) / self.dataloader.batch_size )

    @model.setter
    def model(self, model):
        self.__model = model

    def client_update(self, optimizer, optimizer_args, loss_fn):
        raise NotImplementedError

    def __len__(self):
        return len(self.dataloader.dataset)


class FedAvgClient(Client):
    def client_update(self, optimizer, optimizer_args, loss_fn):
        self.model.train()
        self.model.to(self.device)
        optimizer = optimizer(self.model.parameters(), **optimizer_args)
        
        epochs_to_perform = self.local_epoch
        
        if self.should_use_heterogeneous_E:
            epochs_to_perform = random.randint(self.local_epoch_min, self.local_epoch_max)

        self.epochs_to_perform = epochs_to_perform
        #print("CLIENT %d" % self.client_id)
        for i in range(epochs_to_perform):
            #print("EPOCH %d" % i)
            for img, target in self.dataloader:
                img = img.to(self.device)
                target = target.to(self.device)
                optimizer.zero_grad()
                logits = self.model(img)
                loss = loss_fn(logits, target)
                #print(loss.item())
                loss.backward()
                optimizer.step()

        # evaluate loss and accurary after training finished (for tracking total training loss/accuracy)
        self.model.eval()
        train_loss = 0
        train_correct = 0
        for img, target in self.dataloader:
            img = img.to(self.device)
            target = target.to(self.device)
            optimizer.zero_grad()
            logits = self.model(img)
            loss = loss_fn(logits, target)
            train_loss += loss.item()
            pred = logits.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
        
        self.most_recent_avg_loss = train_loss/len(self.dataloader)
        self.most_recent_num_correct = train_correct

        self.model.to("cpu")