import numpy as np
import torch 
import torch.utils
import torch.utils.data
import torchvision #store the common dataset used in the computer vision

import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,)) # normalize the image, the numbers are mean and sd
])
batch_size_train = 64
batch_size_test = 1000
dataset_dir = "/home/yuzhen/Desktop/ml_learn/number_detection_project/dataset"
train_dataset = torchvision.datasets.MNIST(dataset_dir, train = True, download= True, transform=transform)
train_loader = torch.utils.data.DataLoader (train_dataset,batch_size = batch_size_train, shuffle= True)


test_dataset = torchvision.datasets.MNIST(dataset_dir, train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size_test,shuffle=True)