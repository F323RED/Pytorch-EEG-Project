import torch
from torchvision import transforms
import torch.nn.functional as F 

import os
import shutil
from tqdm import tqdm
from PIL import Image

# Configure logging at the root level of Lightning
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.FATAL)

# Suppress all warning. They are so annoying.
import warnings
warnings.filterwarnings("ignore")


# Main function
if __name__ == '__main__':
    # ------------------------------------------------------------------------ #
    # Misc settings
    VERSION = 1.0
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


    # Which model to use
    # modelName = "EEGNet"
    modelName = "EEGConformer"

    # This decide which subject's data to load
    subjectName = "fred"
    # subjectName = "charli"
    # subjectName = "eddi"

    # ------------------------------------------------------------------------ #
    

    # Path setting
    currentWorkingDir = os.getcwd()
    modelSaveDir = os.path.join(currentWorkingDir, "models")
    modelPath = os.path.join(modelSaveDir, f"{modelName}-{subjectName}-v{VERSION}.pt")


    # Load existing model
    print("Loading model...")
    if os.path.exists(modelPath) :
        with open(modelPath, mode='rb') as f:
            model = torch.jit.load(f)
            model.eval()
            model.to(DEVICE)

        print(f"Model is ready and running on {DEVICE}.")
    else :
        print(f"{modelPath} not found.")
        exit()
    print()