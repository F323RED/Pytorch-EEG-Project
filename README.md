Pytorch EEG Project
===

What is this.
---
Graduation project. This project aims to assist people with motor neuron disease, a condition that affects the nerves that control voluntary muscles. We use machine learning techniques to analyze their electroencephalogram (EEG) signals and infer their intended movements.  
Then, we map these movements to a virtual reality environment, where they can interact with objects and other users. This way, we hope to enhance their quality of life and social connection through a novel application of the metaverse concept.

Enviroment
---
This project using PyTorch 2.0 and PyTorch Lightning framework to accelerate training process.  
Need CUDA 11.8 and Anaconda installed.  Open terminal and execute following command.  
```
conda env create --name ENV_NAME -f /path/to/environment.yml
```

Files explain
---
* **ModelTrain.py**  
  This script is to train the model. The user can tweak some parameters in the setting area, such as the learning rate, the number of epochs, and the batch size. The script will output the trained model and some evaluation metrics after the training is done.  
  
* **EEGDataset.py**  
  This module implement a EEGDataset inhert from PyTorch Dataset. It will load dataset file under ./data/ and do some preprocessing. It provides an interface for accessing EEG data. The preprocessing steps include normalization and convert to torch tensor.
  
* **EEGNetModel.py**  
  EEGNet model is defined here. EEGNet is a deep learning model for electroencephalography (EEG) signal classification. EEGNet is designed to be compact, efficient and generalizable across different EEG tasks and datasets. EEGNet consists of four main layers: a temporal convolution layer, a depthwise convolution layer, a separable convolution layer and a classification layer.
  
* **EEGConformerModel.py**
  EEGConformer model is defined here. EEGConformer is a novel deep learning model for EEG signal analysis. It combines the advantages of convolutional neural networks (CNNs) and transformer networks to capture both spatial and temporal features of EEG signals.
  
* **brain_gui_realtime_project.py**  
  This program is a bridge between different components of this project. It establishes a Bluetooth connection with a brain-computer interface (BCI) device that get EEG signals from the user's brain. It then processes the received signals and passes them to a trained model that predicts the user's intention. The predicted result is then sent to Unity, a game engine that allows the user to control a character in virtual reality (VR).  
  This program require **eeg_decoder.py** and **Ui_NewGUI.py** to run.
  
 How to use
 ---
 1. Make sure EEG dataset is located in ./data/
 2. Set training parameters in **ModelTrain.py**. Such as learning rate, batchsize, subject and which model (EEGNet/EEGConformer) to use.
 3. Start training.
 4. Trained model is located in ./models/
 5. Start **brain_gui_realtime_project.py**
 6. Press ‘connect’ to establish a connection with the BCI device.
 7. Wait for the connection to complete and start the VR.
