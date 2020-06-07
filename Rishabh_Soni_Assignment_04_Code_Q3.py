"""CNN on CIFAR single layer"""

import torch
import pickle
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import product
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

torch.backends.cudnn.enabled = False

seed = 2019

print(torch.__version__)

torch.manual_seed(seed)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='E:/Documents/DLIISC/cifar/', 
                                        train=True,
                                        download=True, 
                                        transform=transform)

traindata, temp = torch.utils.data.random_split(trainset, [10000,40000])

validata, temp = torch.utils.data.random_split(temp, [2000,38000])

del temp

testset = torchvision.datasets.CIFAR10(root='E:/Documents/DLIISC/cifar/', 
                                       train=False,
                                       download=True, 
                                       transform=transform)

testdata, temp = torch.utils.data.random_split(testset, [2000,8000])

del temp

def loader(traindata,validata,testdata,bs):
    trainloader = torch.utils.data.DataLoader(traindata, 
                                          batch_size=bs,
                                          shuffle=True)
    
    validloader = torch.utils.data.DataLoader(validata, 
                                          batch_size=bs,
                                          shuffle=True)
    
    testloader = torch.utils.data.DataLoader(testdata, 
                                         batch_size=bs,
                                         shuffle=False)
    
    return trainloader, validloader, testloader

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3)
        self.fc1 = nn.Linear(8*15*15, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return F.log_softmax(x,dim=1)

def CIFAR(bs,params,epochs,seed):   
    train_loader, valid_loader, test_loader = loader(traindata,validata,testset,bs)
    
    network = Net()
    optimizer = optim.Adam(network.parameters(), lr=params, betas=(0.9, 0.999),\
                           eps=1e-08, weight_decay=0, amsgrad=False)
 
    loss_valid = torch.zeros(epochs,dtype=float)
    acc_valid = torch.zeros(epochs,dtype=float)
    
    for i in range(epochs):
        network.train() ## Intimate the model that you are training
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
        network.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                output = network(data)
                loss_valid[i] += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                acc_valid[i] += pred.eq(target.view_as(pred)).sum().item()
        loss_valid[i] /= len(valid_loader.dataset)
        acc_valid[i] /= len(valid_loader.dataset)
        
    network.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            test_acc += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_acc /= len(test_loader.dataset)
           
    return loss_valid, acc_valid, test_loss, test_acc


epochs = 15

counter = 0

learn_rate_lst = list([0.01, 0.05])
batch_size_lst = list([128, 256])

grid_hyperparam = list(product(learn_rate_lst,batch_size_lst))

valid_loss_store = torch.zeros(len(grid_hyperparam),epochs+2,dtype=float)
valid_acc_store = torch.zeros(len(grid_hyperparam),epochs+2,dtype=float)

test_loss_store = torch.zeros(len(grid_hyperparam),3,dtype=float)
test_acc_store = torch.zeros(len(grid_hyperparam),3,dtype=float)
   
for lr, batchsize in grid_hyperparam:
    print("Iteration {} Start".format(counter))
    lv,av,lt,at = CIFAR(batchsize,lr,epochs,seed)
    
    ## Store accuracy and loss for validation data for each epoch
    valid_loss_store[counter,0] = lr
    valid_loss_store[counter,1] = batchsize
    valid_acc_store[counter,0] = lr
    valid_acc_store[counter,1] = batchsize
    valid_loss_store[counter,2:] = lv
    valid_acc_store[counter,2:] = av
    ## Store accuracy and loss for test data
    test_loss_store[counter,0] = lr
    test_loss_store[counter,1] = batchsize
    test_acc_store[counter,0] = lr
    test_acc_store[counter,1] = batchsize
    test_loss_store[counter,2] = lt
    test_acc_store[counter,2] = at
    
    print("Iteration {} Done".format(counter))
                
    counter += 1    
    
fig = plt.figure(figsize=(10,8))
ax = fig.gca(projection='3d')
ax.scatter(xs=test_acc_store[:,1], ys=test_acc_store[:,0], \
                zs=test_acc_store[:,2], cmap = cm.coolwarm, s=30, c=test_acc_store[:,2])
ax.set_xlabel('BatchSize')
ax.set_ylabel('LearningRate')
ax.set_zlabel('Accuracy')
ax.set_title("Test Data Performance")
plt.show()