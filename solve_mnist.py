import os
import sys

import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST
from resnet import ResNet50

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64

if __name__ == '__main__':
    max_epochs = 3
    if len(sys.argv) > 1:
        max_epochs = int(sys.argv[1])
        print(f'Max epochs = {max_epochs}')
    # Init our model
    mnist_model = ResNet50(num_classes=10)

    # Init DataLoader from MNIST Dataset
    train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=8, persistent_workers=True)

    # Initialize a trainer
    trainer = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=3,
        progress_bar_refresh_rate=100,
    )

    # Train the model ⚡
    trainer.fit(mnist_model, train_loader)
