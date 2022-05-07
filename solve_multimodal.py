import os
import argparse

import torch
from pytorch_lightning import Trainer, loggers
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from transformers import BertForSequenceClassification
from torchvision import transforms

from multimodal_model import MultiModalModel
from dataset.fakeddit_dataset import FakedditHybridDataset, my_collate

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Fakeddit.')
    parser.add_argument('--num-epochs', dest='num_epochs', type=int, nargs='?', help='number of training epochs', default=3)
    parser.add_argument(
        '--num-classes', dest='num_classes', type=int, help='number of classes',
        default=2, choices=[2, 3, 6]
    )
    parser.add_argument('--agg-type', dest='agg_type', type=str, help='type of aggregation text and image vectors',
                        default='conc', choices=['conc', 'conc.avg+sum'])
    parser.add_argument('--freeze-type', dest='freeze_type', type=str, help='freeze of text and image models weights',
                        default='', choices=['', 'image', 'text', 'image+text'])
    args = parser.parse_args()
    freeze = set(args.freeze_type.split('+'))
    freeze_image_model, freeze_text_model = 'image' in freeze, 'text' in freeze
    print(f'Max epochs = {args.num_epochs}, num_classes = {args.num_classes}')

    # Init our model
    bert = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    multimodal_model = MultiModalModel(text_model=bert, num_classes=args.num_classes, agg_type=args.agg_type,
                                       freeze_text_model=freeze_text_model, freeze_image_model=freeze_image_model)

    # Init Dataset
    parts = ['train', 'validate']
    dataset_dir = './dataset'
    image_dir = './dataset/images/'
    datasets_parts = {'train': 'multimodal_train.tsv', 'validate': 'multimodal_validate.tsv'}
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    datasets = {
        part: FakedditHybridDataset(os.path.join(dataset_dir, datasets_parts[part]),
                                    image_dir, img_transform=img_transform)
        for part in parts
    }
    dataloaders = {part: DataLoader(datasets[part], batch_size=64, shuffle=part=='train', num_workers=4,
                                    persistent_workers=True, collate_fn=my_collate)
                   for part in parts}

    # Initialize a trainer
    trainer = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=args.num_epochs,
        progress_bar_refresh_rate=5,
        logger=loggers.TensorBoardLogger('logs/'),
    )
    # Train the model ⚡
    trainer.fit(multimodal_model, dataloaders['train'], dataloaders['validate'])
