import numpy as py
import model 
# from mnist_number_detection import model_save_dir, test_loader
import torch 
import torchvision
from torchvision.transforms import transforms
dataset_dir = "/home/yuzhen/Desktop/ml_learn/number_detection_project/dataset"
model_save_dir = "/home/yuzhen/Desktop/ml_learn/number_detection_project/md.pth"
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,)) # normalize the image, the numbers are mean and sd
])
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.001

test_dataset = torchvision.datasets.MNIST(dataset_dir, train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size_test,shuffle=True)

## initialize the model from the model file 
network = model.cnn_model()
## feed in the parameters to the model from the pth file 
network.load_state_dict(torch.load(model_save_dir))
network.eval()

correct_prediction = 0
wrong_prediction = 0
print("the size of the test_dataset is: ", len(test_loader.dataset))
for batch_index, (data,target) in enumerate(test_loader):
    for single_data, single_target in zip(data,target):
        
        img = single_data
        img = img.unsqueeze(0)
        result = network(img)
        pred = int(torch.argmax(result).item())
        # print("the ground truth is: ",single_target)
        # print("the prediction is: ", pred)
        if single_target.item() == pred:
            correct_prediction+=1
        else:
            wrong_prediction+=1


total_prediction = correct_prediction + wrong_prediction
percentage = (correct_prediction / total_prediction) * 100
print("the correct_prediction num is:", correct_prediction)
print("the wrong_prediction num is: ", wrong_prediction)
print("the total_prediction num is: ", total_prediction)
print(f"the correct rate is: {percentage:.2f}%")


