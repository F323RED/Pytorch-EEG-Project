import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger

import os
import datetime

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
    # ------------------------All settings are here---------------------------- #
    VERSION = 0.1
    
    # Hyper parameters
    LEARNING_RATE = 0.001
    MAX_EPOCHS = 300
    BATCH_SIZE = 100

    # Which model to use
    # modelName = "EEGNet"
    modelName = "EEGConformer"
    

    # This decide which subject's data to load
    subjectName = "fred"
    # subjectName = "charli"
    # subjectName = "eddi"
    # ------------------------------------------------------------------------ #

    # Date record
    now = datetime.datetime.now()
    timeStamp = f"{now.year}{now.month:>02d}{now.day:>02d}{now.hour:>02d}{now.minute:>02d}"

    # Path setting
    currentWorkingDir = os.getcwd()
    datasetDir = os.path.join(currentWorkingDir, "data", subjectName)
    modelSaveDir = os.path.join(currentWorkingDir, "params")
    modelSavePath = os.path.join(modelSaveDir, f"{modelName}-{subjectName}-v{VERSION}-{timeStamp}.pt")
    checkPointDir = os.path.join(currentWorkingDir, "checkpoint")
    logDir = os.path.join(currentWorkingDir, "logs")
    logName = f"{modelName}-{subjectName}-v{VERSION}-{timeStamp}"


    # Create folder when needed
    if not os.path.isdir(datasetDir) :
        os.mkdir(datasetDir)
        print(f"\"{datasetDir}\" created.")
        print()

    if not os.path.isdir(modelSaveDir) :
        os.mkdir(modelSaveDir)
        print(f"\"{modelSaveDir}\" created.")
        print()


    # Load EEG dataset. 
    # Shape: (num, channel, data points) = (num, 8, 500)
    print("Load train dataset")
    trainDataset = EEGDataset(root=datasetDir, type="train")

    print("Load test dataset")
    testDataset = EEGDataset(root=datasetDir, type="test")
    
    print("Load val dataset")
    valDataset = EEGDataset(root=datasetDir, type="val")


    # DataLoader settings
    # Keep this 0 unless you know what you are doing.
    NUM_WORKERS = 0
    PIN_MEM = True
    PERSIS_WORKER = (NUM_WORKERS > 0)

    trainDataLoader = DataLoader(trainDataset,
                                 batch_size=BATCH_SIZE,
                                 shuffle=True,
                                 drop_last=True,
                                 num_workers=NUM_WORKERS,
                                 persistent_workers=PERSIS_WORKER,
                                 pin_memory=PIN_MEM)

    testDataLoader = DataLoader(testDataset,
                                batch_size=BATCH_SIZE,
                                drop_last=True,
                                pin_memory=PIN_MEM)
    
    valDataLoader = DataLoader(valDataset,
                               batch_size=BATCH_SIZE,
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


    # Load model.
    if modelName == "EEGNet" :
        model = EEGNet(lr=LEARNING_RATE)
    elif modelName == "EEGConformer" :
        model = EEGConformer(lr=LEARNING_RATE)
    else :
        print("No model found.")
        exit()


    # CSV logger. It will save metric as metric.csv file.
    logger = CSVLogger(logDir, name=logName, version="")

    # pytorch-lightning trainer
    trainer = Trainer(max_epochs=MAX_EPOCHS,
                      accelerator="auto",
                      check_val_every_n_epoch=5,
                      log_every_n_steps=10,
                      logger=logger,
                      default_root_dir=checkPointDir,
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
    print()


    # Print model training result
    print("─" * 80)
    subjectName = datasetDir.split('\\')[-1]
    print(f"Subject: {subjectName}")
    print()

    print("─" * 80)
    print("Model parameters:")
    parameterDict = model.GetModelParameters()
    for key in parameterDict.keys() :
        print(f"{key}: {parameterDict[key]}")
    print()

    print("─" * 80)
    print("Model metrics: ")
    model.PrintAndResetTestMetrics()

    print("─" * 80)
    print()


    # Save model
    while(True) :
        if os.path.exists(modelSavePath):
            answer = input("Found existing model. Do you want to overwrite it? (yes/no): ")
        else :
            answer = input("Do you want to save the model? (yes/no): ")

        if(answer.lower() == "yes" or answer.lower() == "y") :
            with open(modelSavePath, mode='wb') as f:
                torch.save(model.state_dict(), f)
                print("Model saved.")
            break

        elif(answer.lower() == "no" or answer.lower() == "n") :
            break
    print()