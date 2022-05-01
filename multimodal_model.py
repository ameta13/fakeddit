from resnet import ResNet50
from transformers import BertForSequenceClassification
from pytorch_lightning import LightningModule
from torch import optim, nn
import torch
from torch.nn import functional as F


class MultiModalModel(LightningModule):
    def __init__(self, num_classes: int = 2, image_model=ResNet50, text_model=BertForSequenceClassification):
        super(MultiModalModel, self).__init__()
        self.num_classes = num_classes if num_classes > 2 else 1
        # Image model
        self.image_model = image_model()
        image_model_feature_size = self.image_model.fc.in_features
        self.image_model.fc = nn.Linear(image_model_feature_size, image_model_feature_size)
        # Freeze image model
        for param in self.image_model.parameters():
            param.requires_grad = False

        # Text model
        self.text_model = text_model.bert
        text_model_feature_size = text_model.classifier.in_features
        self.text_model.eval()
        # Freeze text model
        for param in self.text_model.parameters():
            param.requires_grad = False

        # Last layer
        self.linear = nn.Linear(text_model_feature_size + image_model_feature_size, num_classes)

        # Set train parameters
        self.lr = 1e-4

    def forward(self, batch_in: dict):
        text_vector = self.text_model(
            batch_in['bert_input_id'].squeeze(),
            attention_mask=batch_in['bert_attention_mask'].squeeze()
        ).pooler_output
        image_vector = self.image_model(batch_in['image'])
        x = torch.cat((text_vector, image_vector), dim=1)
        x = self.linear(x)
        return x

    def common_step(self, batch, batch_idx, mode):
        # x, y = batch
        # y_pred = self(x)
        # loss = F.cross_entropy(y_pred, y)  # F.mse_loss(self(x), x)
        # self.log(f"{mode}_loss", loss)
        # assert len(y.shape) == 1
        # self.log(f"{mode}_accuracy", torch.sum(y_pred.max(axis=1).indices == y) / y.shape[0])
        # return loss
        x = {
        'bert_input_id': batch['bert_input_id'], 
        'bert_attention_mask': batch['bert_attention_mask'], 
        'image': batch['image']
        }
        y = batch['label']
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)  # F.mse_loss(self(x), x)
        self.log(f"{mode}_loss", loss)
        assert len(y.shape) == 1
        self.log(f"{mode}_accuracy", torch.sum(y_pred.max(axis=1).indices == y)/y.shape[0])
        return loss

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        return self.common_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, 'validation')

    def test_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
