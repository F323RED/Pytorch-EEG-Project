import torch
import torch.optim.lr_scheduler as ls
import torchvision.transforms as transforms
import torch.nn.functional as F

import pytorch_lightning as pl
import torchmetrics

# Design model
class EEGNet(pl.LightningModule) :
    def __init__(self, lr=0.001) :
        super(EEGNet, self).__init__()

        # Define some model parameters
        NUM_CHANNEL = 8
        TIME_POINT = 500
        SAMPLE_RATE = TIME_POINT // 2

        NUM_TEMPORAL_FILTER = 16
        NUM_SPATIAL_FILTER = 16

        DROPOUT_RATE = 0.25
        NUM_CLASS = 5

        self.lr = lr

        # Define model metrics
        self.metric = torchmetrics.Accuracy(task="multiclass", num_classes=NUM_CLASS, average="macro")


        # Define model
        # Block 1 
        # input: (N, 1, CHANNEL, TIME_POINT)
        self.convBlock1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, NUM_TEMPORAL_FILTER, (NUM_CHANNEL, 1), padding=0),
            torch.nn.BatchNorm2d(NUM_TEMPORAL_FILTER),
            torch.nn.ReLU(),
            torch.nn.Dropout(DROPOUT_RATE)
        )

        self.zeroPadding1 = torch.nn.ZeroPad2d((16, 17, 0, 1))
        
        # Block 2
        # input: (N, 1, 16, TIME_POINT)
        self.convBlock2 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, (2, 32), padding=0),
            torch.nn.BatchNorm2d(4),
            torch.nn.MaxPool2d((2, 4)),
            torch.nn.ReLU(),
            torch.nn.Dropout(DROPOUT_RATE)
        )

        self.zeroPadding2 = torch.nn.ZeroPad2d((2, 1, 3, 4))

        # Block 3
        # input: (N, 4, 8, TIME_POINT/4)
        self.convBlock3 = torch.nn.Sequential(
            torch.nn.Conv2d(4, 4, (8, 4), padding=0),
            torch.nn.BatchNorm2d(4),
            torch.nn.MaxPool2d((2, 4), padding=(0, 2)),
            torch.nn.ReLU(),
            torch.nn.Dropout(DROPOUT_RATE)
        )

        # Block 4
        # input: (N, 4, 4, TIME_POINT/16)
        self.linearBlock1 = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(4*4*32, NUM_CLASS)
        )

    def forward(self, x) :
        pred = self.convBlock1(x)
        pred = torch.permute(pred, (0, 2, 1, 3))

        pred = self.zeroPadding1(pred)
        pred = self.convBlock2(pred)

        pred = self.zeroPadding2(pred)
        pred = self.convBlock3(pred)

        pred = self.linearBlock1(pred)

        return pred

    def training_step(self, batch, batch_idx) :
        x, y = batch

        pred = self(x)
        loss = F.cross_entropy(pred, y)

        target = torch.argmax(y, dim=1)
        result = torch.argmax(pred, dim=1)

        # Collect metrics
        accuracy = self.metric(target, result)
        self.log("train_acc", accuracy, prog_bar=True, on_epoch=True)
        self.log("train_loss", loss)

        return loss

    def test_step(self, batch, batch_idx) :
        x, y = batch

        with torch.no_grad() :
            pred = self(x)
            loss = F.cross_entropy(pred, y)

        target = torch.argmax(y, dim=1)
        result = torch.argmax(pred, dim=1)

        # Collect metrics
        accuracy = self.metric(target, result)
        self.log("test_acc", accuracy)
        self.log("test_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx) :
        x, y = batch

        with torch.no_grad() :
            pred = self(x)
            loss = F.cross_entropy(pred, y)

        target = torch.argmax(y, dim=1)
        result = torch.argmax(pred, dim=1)

        # Collect metrics
        accuracy = self.metric(target, result)
        self.log("val_acc", accuracy)
        self.log("val_loss", loss)

        return loss
    
    def configure_optimizers(self) :
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ls.ExponentialLR(optimizer, gamma=0.9),
                "interval": "epoch",
                "frequency": 1,
            },
        }