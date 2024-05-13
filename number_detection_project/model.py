import numpy as py 
import torch.nn as nn 
import torch.nn.functional as F

## when we define a model:
# 1. create model class 
#       -> define the structure of the model
#       -> define the forward path 
# 2. define the optimizer 
#       --> optimizer is used to adjust the parameters in the model 
# 3. loss function
#       --> the optimizer will try to optimize the model in the opposite direction of the loss function

class cnn_model(nn.Module):
    def __init__(self):
        super(cnn_model,self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=3, stride=1, padding=1) # the output is 28*28*10
        self.conv2 = nn.Conv2d(10,20,kernel_size=3, stride= 1, padding=1) # the output is 28*28*20
        self.drop_out = nn.Dropout2d(p = 0.5)
        self.fc1 = nn.Linear(7 * 7 * 20, 1280)
        self.fc2 = nn.Linear(1280,10)
        self.max_pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.relu = nn.LeakyReLU()

    def conv_combine(self,x,current_conv,if_use_drop_down):
        if if_use_drop_down == False:
            x = current_conv(x)
            x = self.max_pool(x)
            x = self.relu(x)
        else: 
            x = current_conv(x)
            x = self.drop_out(x) # randomly zero 50 % of the input tensor 
            x = self.max_pool(x)
            x = self.relu(x)
        return x

    def forward(self,x):
        x = self.conv_combine(x,self.conv1,False)
        x = self.conv_combine(x,self.conv2,True)
        x = x.view(x.size(0),-1) # the first parameter is the batch size 
        if self.training == True:
            x = self.relu(self.drop_out(self.fc1(x)))
        else:    
            x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x,dim=1)
        return x 

