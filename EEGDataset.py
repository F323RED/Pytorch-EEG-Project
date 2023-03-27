import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms

import os
import numpy as np


# Custom pytorch dataset
class EEGDataset(Dataset):
    def __init__(self, root, type="train"):
        # Hyper parameter
        NUM_CLASS = 5

        # Check if dataset type is valid.
        VALID_TYPE = ["train", "test", "val"]
        if not type in VALID_TYPE :
            raise BaseException("Invalid dataset type. 'type' should one of follow: 'train', 'test', 'val'.")

        dataPath = os.path.join(root, f"x_{type}.npy")
        labelPath = os.path.join(root, f"y_{type}.npy")


        # Create transformer for data pre-processing
        # This apply z-score normalize
        # Mean and std comes from all EEG data across all subject
        preprocessor = transforms.Normalize(mean=(-6.67145e-10), std=(1.63550e-05))


        # Check if dataset file exist.
        if not os.path.isfile(dataPath) or not os.path.isfile(labelPath) :
            print(dataPath, "or", labelPath, "is missing!")
            exit()
        
        
        # Load EEG data and apply pre-processing
        self.x = torch.from_numpy(np.load(dataPath))
        self.x = self.x.type(torch.float32)
        self.x = self.x.view((self.x.shape[0], 1, self.x.shape[1], self.x.shape[2]))

        # Create transformer for data pre-processing
        # This apply z-score normalize
        # Mean and std comes from all EEG data across all subject
        std = self.x.std(dim=(0, 2, 3))
        mean = self.x.mean(dim=(0, 2, 3))
        preprocessor = transforms.Normalize(mean=mean, std=std)
        
        self.x = preprocessor(self.x)

        # Load labels
        self.y = torch.from_numpy(np.load(labelPath))
        self.y = self.y.type(torch.int64)
        self.y = F.one_hot(self.y, NUM_CLASS)
        self.y = self.y.type(torch.float32)

        self.numSamples = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.numSamples