import numpy as np
import PIL.Image as Image
import torch
from utils.data import get_mnist_data
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import random

class MnistLocalDataset(Dataset):
    def __init__(self, images, labels, client_id, model='lr'):
        self.images = images
        self.labels = labels.astype(int)
        self.client_id = client_id
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # Images must be at least 224x224 for VGG
            transforms.Pad(padding=[2,], fill=0, padding_mode='constant'),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])
        self.model = model

    def __getitem__(self, index):
        # VGG requires images with 3 channels
        arg = np.dstack([self.images[index].reshape(28, 28) for _ in range(3)])
        img = Image.fromarray(arg, mode='RGB')

        img = self.transform(img)
        target = self.labels[index]
        return img, target

    def __len__(self):
        return len(self.images)

def create_mnist_datasets(num_clients=100,
                          datadir="./data/mnist",
                          should_use_heterogeneous_data=False,
                          iid=False,
                          model='lr'):
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
        # shard_sizes_old = np.random.power(3, size=num_clients)
        # shard_sizes_old /= shard_sizes_old.sum()
        # shard_sizes_old *= len(train_img)
        # shard_sizes_old = shard_sizes_old.astype(int)
        # assign samples per client using a power law distribution
        # make sure we don't assign more samples to clients than we actually have access to
        min_shard_size = int(.7*(train_img.shape[0] // num_clients))
        while True:
            shard_sizes = min_shard_size + np.random.lognormal(2, 4, num_clients).astype(int)
            if 0.9*train_img.shape[0] <= np.sum(shard_sizes) <= train_img.shape[0]:
                break
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

        local_datasets.append(MnistLocalDataset(img, label, client_id, model))

    test_sorted_index = np.argsort(test_label)
    test_img = test_img[test_sorted_index]
    test_label = test_label[test_sorted_index]

    test_dataset = MnistLocalDataset(test_img, test_label, -1, model)

    # show distribution of classes across clients
    if False:
        for j in range(5):
            class_bins=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])-.5
            f = plt.figure(figsize=(14, 9))
            for i in range(20):
                plt.subplot(4, 5, i+1)
                plt.title('Client {}'.format(20*j+i))
                plt.hist(
                        local_datasets[20*j+i].labels,
                        density=False,
                        bins=class_bins)
            plt.show()

    return local_datasets, test_dataset