import torch

import pytorch_lightning as pl
from pytorch_lightning import Trainer

import os

from EEGNetModel import EEGNet
from EEGDataset import EEGDataset, DataLoaderX


# configure logging at the root level of Lightning
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


# Hyper parameters
learningRate = 0.001
epochs = 300
batchSize = 100

# Misc setting
modelName = "EEGNet_v0.1"
currentPath = os.getcwd()
modelPath = os.path.join(currentPath, "params", modelName + ".pt")

# datasetDir = r"D:\程式碼\Pytorch EEG\data\fred"
datasetDir = r"D:\程式碼\Pytorch EEG\data\charli"
# datasetDir = r"D:\程式碼\Pytorch EEG\data\eddi"

checkPointPath = os.path.join(currentPath, "checkpoint")


# Main function
if __name__ == '__main__':
    # Load EEG dataset. 
    # Shape: (num, channel, data points) = (num, 8, 500)
    print("Load train dataset")
    trainDataset = EEGDataset(root=datasetDir, type="train")

    print("Load test dataset")
    testDataset = EEGDataset(root=datasetDir, type="test")
    
    print("Load val dataset")
    valDataset = EEGDataset(root=datasetDir, type="val")

    
    trainDataLoader = DataLoaderX(trainDataset,
                                  batch_size=batchSize,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=6)

    testDataLoader = DataLoaderX(testDataset,
                                 batch_size=batchSize,
                                 drop_last=True,
                                 num_workers=6)
    
    valDataLoader = DataLoaderX(valDataset,
                                 batch_size=batchSize,
                                 drop_last=True,
                                 num_workers=6)


    print("Done loading dataset.")
    print()

    print("Len of train data:", len(trainDataset))
    print("Len of test data:", len(testDataset))
    print("Len of val data:", len(valDataset))
    print()


    # .load_from_checkpoint("/path/to/checkpoint.ckpt")
    model = EEGNet(lr=learningRate)


    # Load existing model
    if os.path.exists(modelPath) :
        while(True) :
            answer = input("Found existing model. Do you want to load it? (yes/no): ")

            if(answer.lower() == "yes" or answer.lower() == "y") :
                with open(modelPath, mode='rb') as f:
                    model.load_state_dict(torch.load(f))
                break

            elif(answer.lower() == "no" or answer.lower() == "n") :
                break
    print()

    # pytorch-lightning train
    trainer = Trainer(max_epochs=epochs,
                      accelerator="auto",
                      check_val_every_n_epoch=3,
                      log_every_n_steps=10,
                      default_root_dir=checkPointPath,
                      benchmark=True)


    print("Pre-test.")
    trainer.test(model, testDataLoader)
    model.PrintAndResetTestMetrics()
    print()

    print("Training.")
    trainer.fit(model, trainDataLoader, valDataLoader)
    print()

    print("Post-test.")
    trainer.test(model, testDataLoader)
    model.PrintAndResetTestMetrics()
    print()


    # Save model
    while(True) :
        if os.path.exists(modelPath):
            answer = input("Found existing model. Do you want to overwrite it? (yes/no): ")
        else :
            answer = input("Do you want to save the model? (yes/no): ")

        if(answer.lower() == "yes" or answer.lower() == "y") :
            with open(modelPath, mode='wb') as f:
                torch.save(model.state_dict(), f)
                print("Model saved.")
            break

        elif(answer.lower() == "no" or answer.lower() == "n") :
            break
    print()