import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger

import os

from EEGNetModel import EEGNet
from EEGConformerModel import EEGConformer
from EEGDataset import EEGDataset, DataLoaderX

# Configure logging at the root level of Lightning
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.FATAL)

# Surpress all warning. They are so annoying.
import warnings
warnings.filterwarnings("ignore")



# Main function
if __name__ == '__main__':
    # Hyper parameters
    learningRate = 0.001
    epochs = 100
    batchSize = 100

    # Misc setting
    modelName = "EEGConformer_v0.1"
    currentPath = os.getcwd()
    modelPath = os.path.join(currentPath, "params", modelName + ".pt")

    datasetDir = r"D:\程式碼\Pytorch EEG\data\fred"
    # datasetDir = r"D:\程式碼\Pytorch EEG\data\charli"
    # datasetDir = r"D:\程式碼\Pytorch EEG\data\eddi"

    checkPointPath = os.path.join(currentPath, "checkpoint")
    logPath = os.path.join(currentPath, "logs")



    # Load EEG dataset. 
    # Shape: (num, channel, data points) = (num, 8, 500)
    print("Load train dataset")
    trainDataset = EEGDataset(root=datasetDir, type="train")

    print("Load test dataset")
    testDataset = EEGDataset(root=datasetDir, type="test")
    
    print("Load val dataset")
    valDataset = EEGDataset(root=datasetDir, type="val")


    # DataLoader settings
    NUM_WORKERS = 0
    PIN_MEM = True
    PERSIS_WORKER = (NUM_WORKERS > 0)

    trainDataLoader = DataLoader(trainDataset,
                                 batch_size=batchSize,
                                 shuffle=True,
                                 drop_last=True,
                                 num_workers=NUM_WORKERS,
                                 persistent_workers=PERSIS_WORKER,
                                 pin_memory=PIN_MEM)

    testDataLoader = DataLoader(testDataset,
                                batch_size=batchSize,
                                drop_last=True,
                                pin_memory=PIN_MEM)
    
    valDataLoader = DataLoader(valDataset,
                               batch_size=batchSize,
                               drop_last=True,
                               num_workers=NUM_WORKERS,
                               persistent_workers=PERSIS_WORKER,
                               pin_memory=PIN_MEM)


    print("Done loading dataset.")
    print()

    print("Len of train data:", len(trainDataset))
    print("Len of test data:", len(testDataset))
    print("Len of val data:", len(valDataset))
    print()



    # model = EEGNet(lr=learningRate)
    model = EEGConformer(lr=learningRate)


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


    # CSV logger. It will save metric.csv file.
    # logger = TensorBoardLogger(save_dir=logPath)
    logger = CSVLogger(logPath, name=f"{modelName}-log")

    # pytorch-lightning train
    trainer = Trainer(max_epochs=epochs,
                      accelerator="auto",
                      check_val_every_n_epoch=5,
                      log_every_n_steps=10,
                      logger=logger,
                      default_root_dir=checkPointPath,
                      benchmark=True, 
                      num_sanity_val_steps=0)


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