from torchvision.models.resnet import ResNet, Bottleneck
import pytorch_lightning as pl
from torch import optim, nn
from torch.nn import functional as F


class ResNet50(ResNet, pl.LightningModule):
    def __init__(self, num_classes: int = 2):
        pl.LightningModule.__init__(self)
        ResNet.__init__(self, Bottleneck, [3, 4, 6, 3], num_classes=num_classes if num_classes > 2 else 1)
        in_channels = 1
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # def forward(self, x) - define in ResNet

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        loss = F.cross_entropy(self(x), y)  # F.mse_loss(self(x), x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
