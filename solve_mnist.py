import os
import argparse

import torch
from pytorch_lightning import Trainer, loggers
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from resnet import ResNet50

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MNIST.')
    parser.add_argument('--num-epochs', dest='num_epochs', type=int, nargs='?', help='number of training epochs', default=3)
    # parser.add_argument('--sum', dest='accumulate', action='store_const',
    #                     const=sum, default=max,
    #                     help='sum the integers (default: find the max)')
    args = parser.parse_args()
    num_epochs = args.num_epochs

    print(f'Max epochs = {num_epochs}')
    # Init our model
    mnist_model = ResNet50(num_classes=10)

    # Init DataLoader from MNIST Dataset
    dataset = MNIST(PATH_DATASETS, download=True, transform=transforms.ToTensor())
    train, val = random_split(dataset, [55000, 5000])

    train_loader = DataLoader(train, batch_size=BATCH_SIZE, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, num_workers=4, persistent_workers=True)

    # Initialize a trainer
    trainer = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=num_epochs,
        progress_bar_refresh_rate=5,
        logger=loggers.TensorBoardLogger('logs/'),
    )
    # Train the model ⚡
    trainer.fit(mnist_model, train_loader, val_loader)
