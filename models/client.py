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
        # TODO! I think this is the source of the error
        # The len(self.dataloader.dataset) is always 10
        # Tau was a fraction, so we were really scaling up the gradients.
        # I think the fix by Tim will fix this.

        # I can confirm that this method works for the mnist dataset.
        # I'm nearly positive it's somehow broken for the synthetic dataset.
        tau = self.epochs_to_perform * len(self.dataloader.dataset) / self.dataloader.batch_size
        # print('Tau {}: {}, {}, {}, {:.3f}'.format(self.client_id, self.epochs_to_perform, len(self.dataloader.dataset), self.dataloader.batch_size, tau))
        return tau

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
        
        # Decide local epochs to use.
        epochs_to_perform = self.local_epoch
        if self.should_use_heterogeneous_E:
            epochs_to_perform = np.random.randint(self.local_epoch_min, self.local_epoch_max)
        self.epochs_to_perform = epochs_to_perform

        losses = []
        number_correct = []
        for i in range(epochs_to_perform):
            for img, target in self.dataloader:
                optimizer.zero_grad()

                img = img.to(self.device)
                target = target.to(self.device)

                logits = self.model(img)
                loss = loss_fn(logits, target)

                losses.append(loss.item())
                pred = logits.argmax(dim=1, keepdim=True)
                number_correct.append(pred.eq(target.view_as(pred)).sum().item())
                
                loss.backward()
                optimizer.step()

        self.most_recent_avg_loss = np.mean(losses)
        self.most_recent_num_correct = np.mean(number_correct)

        # evaluate loss and accurary after training finished (for tracking total training loss/accuracy)
        # self.model.eval()
        # train_loss = 0
        # train_correct = 0
        # for img, target in self.dataloader:
        #     img = img.to(self.device)
        #     target = target.to(self.device)
        #     optimizer.zero_grad()
        #     logits = self.model(img)
        #     loss = loss_fn(logits, target)
        #     train_loss += loss.item()
        #     pred = logits.argmax(dim=1, keepdim=True)
        #     train_correct += pred.eq(target.view_as(pred)).sum().item()
        
        # self.most_recent_avg_loss = train_loss/len(self.dataloader)
        # self.most_recent_num_correct = train_correct

        self.model.to("cpu")