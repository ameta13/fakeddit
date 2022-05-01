from torchvision.models.resnet import ResNet, Bottleneck
from pytorch_lightning import LightningModule
from torch import optim, nn
import torch
from torch.nn import functional as F


class ResNet50(ResNet, LightningModule):
    def __init__(self, num_classes: int = 2, is_greyscale=False):
        LightningModule.__init__(self)
        ResNet.__init__(self, Bottleneck, [3, 4, 6, 3], num_classes=num_classes if num_classes > 2 else 1)
        if is_greyscale:
            in_channels = 1
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # def forward(self, x) - define in ResNet

    def common_step(self, batch, batch_idx, mode):
        x, y = batch
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

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
