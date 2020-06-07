""" CNN on MNIST with 1 convolution layer """

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from itertools import product
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

torch.backends.cudnn.enabled = False

seed = 2019

print(torch.__version__)

torch.manual_seed(seed)

class DataReader:
    def __init__(self):
        pass
    
    def _load_data(self,location,filename):
        data = torch.load(location + filename)
        if(isinstance(data,tuple) and len(data) == 2):
            ind = torch.randperm(int(data[0].shape[0]/10))
            X = data[0].float()
            self.X = X[ind]
            Y = data[1]
            self.Y = Y[ind]
        else:
            print("Not implemented")
        return self.X, self.Y
    
    def data_processor(self,location,filename,split,bs,*args):
        self._load_data(location,filename)
        self.X = torch.reshape(self.X,(self.X.size(0),1,self.X.size(1),self.X.size(2)))
        if split == True:
            x_train = self.X[0:args[0]]/255
            yt = self.Y[0:args[0]]
            x_valid = self.X[args[0]:len(self.X)]/255
            yv = self.Y[args[0]:len(self.X)]
            train_ds = TensorDataset(x_train, yt) ## for dataloader
            train_dl = DataLoader(train_ds, batch_size=bs, drop_last=False, shuffle=True)
            valid_ds = TensorDataset(x_valid, yv)
            valid_dl = DataLoader(valid_ds, batch_size=bs, drop_last=False, shuffle=True)
            return train_dl, valid_dl
        else:
            x_test = self.X/255
            temp_y = self.Y
            test_ds = TensorDataset(x_test, temp_y)
            test_dl = DataLoader(test_ds, batch_size=bs, drop_last=False, shuffle=False)
            return test_dl

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
         ## this conv layer will have depth of 8 as we have 8 channels in the last layer
         ## the output of each convolution will be added across channels
        self.fc1 = nn.Linear(8*13*13, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return F.log_softmax(x,dim=1)
    
location = 'E:/Documents/DLIISC/'
trainfile = 'training.pt'
testfile = 'test.pt'

def MNIST(location,trainfile,testfile,batchsize,params,epochs,seed):   
    train = DataReader()
    train_loader, valid_loader = \
    train.data_processor(location,trainfile,True,batchsize,5000)
    test = DataReader()
    test_loader = test.data_processor(location,testfile,False,batchsize)
    
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

learn_rate_lst = list([0.001, 0.01, 0.05, 0.1])
batch_size_lst = list([1, 32, 128, 1024])

grid_hyperparam = list(product(learn_rate_lst,batch_size_lst))

valid_loss_store = torch.zeros(len(grid_hyperparam),epochs+2,dtype=float)
valid_acc_store = torch.zeros(len(grid_hyperparam),epochs+2,dtype=float)

test_loss_store = torch.zeros(len(grid_hyperparam),3,dtype=float)
test_acc_store = torch.zeros(len(grid_hyperparam),3,dtype=float)
   
for lr, batchsize in grid_hyperparam:
    print("Iteration {} Start".format(counter))
    lv,av,lt,at = MNIST(location,trainfile,testfile,batchsize,lr,epochs,seed)
    
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