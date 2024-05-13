import numpy as np
import torch 
import torch.utils
import torch.utils.data
import torchvision #store the common dataset used in the computer vision

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import model
import torch.nn.functional as F

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,)) # normalize the image, the numbers are mean and sd
])
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.001
dataset_dir = "/home/yuzhen/Desktop/ml_learn/number_detection_project/dataset"
model_save_dir = "/home/yuzhen/Desktop/ml_learn/number_detection_project/md.pth"
train_dataset = torchvision.datasets.MNIST(dataset_dir, train = True, download= True, transform=transform)
train_loader = torch.utils.data.DataLoader (train_dataset,batch_size = batch_size_train, shuffle= True)

test_dataset = torchvision.datasets.MNIST(dataset_dir, train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size_test,shuffle=True)


##push the data to the GPU------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ("the device currently use is: ", device)
##------------------------------------------------------------------------------------

## this part is used to display the shape of the test data and label -----------------
# examples = enumerate(test_loader)
# batch_idx, (example_data, example_targets) = next (examples)

# print(example_data.shape)
# print(example_targets.shape)
# plt.imshow(example_data[0][0],cmap="gray")
# plt.show()
# print("the label of the first data is: ", example_targets[0])

##------------------------------------------------------------------------------------

##design the model (we will write in the seperate file)

# initialize the structure of the model 
network = model.cnn_model() 

#move the model to the GPU 
network.to(device)

# initialize the optimizer  
# the "momentum" here means we will not only concider the current gradient vecrior, but also the accumulated gradient.
# advantage of the momentum: 
#   1. move in the same direction as the previous iteration
#   2, faster convergence 
#   3. less oscillations more smooth 
optimizer = optim.SGD(params=network.parameters(), lr=learning_rate, momentum=0.9) 

#define the loss function
#since it's a multiclass classification problem and we use the log_softmax() as the return of the moodel 
#We will use the nll_loss as the loss function, the full name is negative log likelyhood loss. 
loss = F.nll_loss


#training loop 

def train(epoch):
    network.train()# show we are in the trainning mode right now 
    for batch_idx, (data, target) in enumerate(train_loader):
        #in order to use GPU to calcuate, move the data and target to the GPU:
        data = data.to(device)
        target = target.to(device)
        
        #STEP1: reset the optimizer:
        optimizer.zero_grad()
        #STEP2: calculate the result: 
        #even if when i define the model i don't consider the batch_size, the pytorch can take care of it automatically 
        output = network(data)
        loss_output = loss(output,target) #calcuate the current loss 
        #STEP3:backward:
        loss_output.backward()
        #STEP4:update all the parameters in the model 
        optimizer.step()

        #print out the trainning effect:
        epoch_number = epoch
        total_data_size =  len(train_loader.dataset)
        finished_data_size = batch_idx * len(data)
        percentage = (finished_data_size / total_data_size) * 100 
        loss_value = loss_output.item()
        
        if batch_idx % 10 == 0:
            print(f"current epoch: {epoch_number} ({finished_data_size} / {total_data_size}) {percentage:.2f}%  loss is: {loss_value:.4f}")
 
 
# try to run 3 epoch here: 
for epoch in range(0,6):
    if epoch > 3:
        learning_rate *= 0.1
    train(epoch)


## save the model parameters in the direction so that we can directly use it later 
torch.save(network.state_dict(), model_save_dir)  


