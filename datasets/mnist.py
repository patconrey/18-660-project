import numpy as np
import PIL.Image as Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MnistLocalDataset(Dataset):
    def __init__(self, images, labels, client_id):
        self.images = images
        self.labels = labels.astype(int)
        self.client_id = client_id
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # Images must be at least 224x224 for VGG
            transforms.Pad(padding=[2,], fill=0, padding_mode='constant'),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])

    def __getitem__(self, index):
        # VGG requires images with 3 channels
        arg = np.dstack([self.images[index].reshape(28, 28) for _ in range(3)])
        img = Image.fromarray(arg, mode='RGB')

        img = self.transform(img)
        target = self.labels[index]
        return img, target

    def __len__(self):
        return len(self.images)
