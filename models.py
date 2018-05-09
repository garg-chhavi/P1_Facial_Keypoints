## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 4) #224x224
        self.maxpool = nn.MaxPool2d(2,2) # 112X112
        self.conv2 = nn.Conv2d(32,64,3) #112X112
        self.conv3 = nn.Conv2d(64,128,2) 
        self.conv4 = nn.Conv2d(128,256,1) 
        self.fc1 = nn.Linear(256*13*13,544)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.fc1_drop = nn.Dropout(p=0.4)
        self.output = nn.Linear(544,136)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv1(x)) #224X224X32
        x = self.maxpool(x) #112X112 X32
        x = self.fc1_drop(x)
        x = F.relu(self.conv2(x)) #112x112X64
        x = self.maxpool(x) #54X54X64
        x = self.fc1_drop(x)
        x = F.relu(self.conv3(x)) #54X54X128
        x = self.maxpool(x) #26X26X128
        x = self.fc1_drop(x)
        x = F.relu(self.conv4(x)) #26X26X256
        x = self.maxpool(x) #12X12X256
        x = self.fc1_drop(x)
        
        #print(x.shape) # 10X 26X26X128
        x = x.view(x.size(0), -1) #  batchsizex (128*26*26)
        #print(x.shape)
        x = F.relu(self.fc1(x)) #batchsizex544
        x = self.fc1_drop(x)
        x = self.output(x) #batch sizex 136
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
