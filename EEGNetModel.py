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

        NUM_TEMP_F = 16     # Number of temporal filter
        DEPTH = 4           # Number of spatial filter
        NUM_POINT_F = 4     # Number of point wise filter

        DROPOUT_RATE = 0.25
        NUM_CLASS = 5

        self.lr = lr


        # Model parameter dict
        self.__modelParameters = {
            "num_channel" : NUM_CHANNEL,
            "num_time_point" : TIME_POINT,
            "dropout_rate" : DROPOUT_RATE,
            "lr" : lr,
            "num_temporal_filter" : NUM_TEMP_F,
            "deep_wise_conv_depth" : DEPTH,
            "num_point_wise_filter" : NUM_POINT_F,
            "num_class" : NUM_CLASS
        }


        # Init model metrics
        self.trainAccuracy = torchmetrics.Accuracy(task="multiclass", 
                                              num_classes=NUM_CLASS, 
                                              average="macro")
        
        self.testAccuracy = torchmetrics.Accuracy(task="multiclass", 
                                              num_classes=NUM_CLASS, 
                                              average="macro")
        
        self.valAccuracy = torchmetrics.Accuracy(task="multiclass", 
                                              num_classes=NUM_CLASS, 
                                              average="macro")
        
        self.confMatrix = torchmetrics.ConfusionMatrix(task="multiclass", 
                                                       num_classes=NUM_CLASS)

        self.F1Score = torchmetrics.F1Score(task="multiclass", num_classes=NUM_CLASS)
        self.preci = torchmetrics.Precision(task="multiclass", num_classes=NUM_CLASS)
        self.recall = torchmetrics.Recall(task="multiclass", num_classes=NUM_CLASS)


        # Define model
        # Block 1 
        # input: (N, 1, CHANNEL, TIME_POINT)
        self.convBlock1 = torch.nn.Sequential(
            # This conv2d server as band-pass filter
            torch.nn.Conv2d(1, NUM_TEMP_F, (1, SAMPLE_RATE), padding="same"),
            torch.nn.BatchNorm2d(NUM_TEMP_F),
            # Deep-wise conv2d spatial filter
            torch.nn.Conv2d(NUM_TEMP_F, NUM_TEMP_F * DEPTH, (NUM_CHANNEL, 1), 
                            padding="valid",
                            groups=NUM_TEMP_F),
            torch.nn.BatchNorm2d(NUM_TEMP_F * DEPTH),
            torch.nn.ELU(),
            torch.nn.AvgPool2d((1, 4)),
            torch.nn.Dropout(DROPOUT_RATE)
        )
        
        # Block 2
        # input: (N, NUM_TEMP_F * NUM_SPAT_F, 1, TIME_POINT // 4)
        self.convBlock2 = torch.nn.Sequential(
            # This is suppose to be a separable conv2d. 
            # But I am too lazy to implement this.
            torch.nn.Conv2d(NUM_TEMP_F * DEPTH, NUM_POINT_F, (1, 16), padding="same"),
            torch.nn.BatchNorm2d(NUM_POINT_F),
            torch.nn.ELU(),
            torch.nn.AvgPool2d((1, 8), padding=(0, 4)),
            torch.nn.Dropout(DROPOUT_RATE)
        )

        # Block 3
        # input: (N, NUM_POINT_F, 1, TIME_POINT // 32)
        self.linearBlock1 = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(NUM_POINT_F * 16, NUM_CLASS)
        )

    def forward(self, x) :
        pred = self.convBlock1(x)
        pred = self.convBlock2(pred)

        pred = self.linearBlock1(pred)

        return pred

    def training_step(self, batch, batch_idx) :
        x, y = batch

        pred = self(x)
        loss = F.cross_entropy(pred, y)

        target = torch.argmax(y, dim=1)
        result = torch.argmax(pred, dim=1)

        # Collect metrics
        accuracy = self.trainAccuracy(target, result)
        self.log("train_acc", accuracy, prog_bar=True, on_step=False,on_epoch=True)
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
        accuracy = self.testAccuracy(target, result)
        self.confMatrix.update(target, result)
        self.F1Score.update(target, result)
        self.preci.update(target, result)
        self.recall.update(target, result)

        self.log("test_acc", accuracy, on_step=False, on_epoch=True)
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
        accuracy = self.valAccuracy(target, result)
        self.log("val_acc", accuracy, prog_bar=True, on_step=False, on_epoch=True)
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
    
    def PrintAndResetTestMetrics(self) :
        acc = self.testAccuracy.compute().item() * 100
        f1 = self.F1Score.compute().item()
        pre = self.preci.compute().item()
        rec = self.recall.compute().item()

        # Print accuracy
        print(f"Accuracy: {acc:.02f}%")

        # Print F1 score
        print(f"F1-score: {f1:.03f}")

        # Print precision
        print(f"Precision: {pre:.03f}")
        
        # Print recall
        print(f"Recall: {rec:.03f}")
        print()
        
        # Print confusion matrix
        tempMat = self.confMatrix.compute()
        print("Confusion matrix:")
        for row in tempMat :
            print("|", end=" ")
            for col in row :
                print(f"{col:>3d}", end=" ")
            print("|")

        self.testAccuracy.reset()
        self.confMatrix.reset()
        self.F1Score.reset()
        self.preci.reset()
        self.recall.reset()

    def GetModelParameters(self) :
        return self.__modelParameters