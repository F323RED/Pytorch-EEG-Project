import torch
import torch.optim.lr_scheduler as ls
import torch.nn.functional as F

import pytorch_lightning as pl
import torchmetrics

import einops
from einops.layers.torch import Rearrange, Reduce

# Design model
class EEGConformer(pl.LightningModule) :
    def __init__(self, lr=0.001) :
        super(EEGConformer, self).__init__()

        # Define some model parameters
        # Channel of electrode
        NUM_CHANNEL = 8

        # This affect most dropout layers in the model
        DROPOUT_RATE = 0.5

        # Number of temporal filter
        NUM_TEMP_F = 32

        # Need to convert EED data to (N, L, E) format.
        # So it can fit into TransformerDecoder
        EMBED_SIZE = 40

        # To determine size of feed forward layer of TransformerDecoder
        FEEDFORWARD_EXPANSION = 4

        # This determine how many TransformerDecoderLayer in TransformerDecoder
        # According to the paper, depth of 1 is just enough.
        TRANSFORMER_DEPTH = 1

        # Number of attention head
        NUM_HEAD = 8

        # Linear block setting
        HIDDEN_LAYER_SIZE1 = 256
        HIDDEN_LAYER_SIZE2 = 32
        NUM_CLASS = 5

        self.lr = lr

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
        # Patch embedding. Convert 2D 'image' to serial of tokens.
        # input (N, 1, NUM_CHANNEL, TIME_POINT)
        self.patchEmbeddingBlock = torch.nn.Sequential(
            torch.nn.Conv2d(1, NUM_TEMP_F, (1, 25), padding="same"),
            torch.nn.Conv2d(NUM_TEMP_F, NUM_TEMP_F, (NUM_CHANNEL, 1), padding="valid"),
            torch.nn.BatchNorm2d(NUM_TEMP_F),
            torch.nn.ELU(),
            # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            torch.nn.AvgPool2d((1, 75), (1, 15)),
            torch.nn.Dropout(DROPOUT_RATE)
        )

        # Some sort of feature space projection?
        self.projectionBlock = torch.nn.Sequential(
            torch.nn.Conv2d(NUM_TEMP_F, EMBED_SIZE, (1, 1)),
            Rearrange("b e (h) (w) -> b (h w) e")
        )

        # Encoder layer. Using multi-head attention.
        self.transformerEncoderLayer = torch.nn.TransformerEncoderLayer(
            EMBED_SIZE,
            NUM_HEAD,
            EMBED_SIZE * FEEDFORWARD_EXPANSION,
            dropout=DROPOUT_RATE,
            activation=torch.nn.ELU(),
            batch_first=True
        )

        # A transformer encoder.
        # NOTE: This didn't have residue & norm layer. So it might be hard to train.
        self.transformerEncoder = torch.nn.TransformerEncoder(
            self.transformerEncoderLayer,
            TRANSFORMER_DEPTH,
        )

        # Linear block. Output classes.
        self.linearBlock = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.LayerNorm(EMBED_SIZE * 29),
            torch.nn.Linear(EMBED_SIZE * 29, HIDDEN_LAYER_SIZE1),
            torch.nn.ELU(),
            torch.nn.Dropout(DROPOUT_RATE),
            torch.nn.Linear(HIDDEN_LAYER_SIZE1, HIDDEN_LAYER_SIZE2),
            torch.nn.ELU(),
            torch.nn.Dropout(DROPOUT_RATE),
            torch.nn.Linear(HIDDEN_LAYER_SIZE2, NUM_CLASS)
        )


    def forward(self, x) :
        pred = self.patchEmbeddingBlock(x)
        pred = self.projectionBlock(pred)

        pred = self.transformerEncoder(pred)

        pred = self.linearBlock(pred)

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

