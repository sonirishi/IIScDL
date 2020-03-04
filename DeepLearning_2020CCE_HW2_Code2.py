# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 23:26:41 2020

@author: Admin
"""

import torch
from numpy import arange
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import matplotlib.pyplot as plt
from itertools import product
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

seed = 2019

torch.manual_seed(seed)

## Dataset is abstract class

class DataReader:
    def __init__(self):
        pass
    
    def _load_data(self,location,filename):
        data = torch.load(location + filename)
        if(isinstance(data,tuple) and len(data) == 2):
            self.X = data[0].float()
            self.Y = data[1]
        else:
            print("Not implemented")
        return self.X, self.Y
    
    def _onehotencoder(self,tensor):
        y_train = torch.zeros((len(tensor),len(tensor.unique())))
        y_train[arange(len(y_train)),tensor] = 1
        return y_train
    
    def data_processor(self,location,filename,split,bs,*args):
        self._load_data(location,filename)
        self.X = torch.reshape(self.X,(self.X.size(0),self.X.size(1)*self.X.size(2)))
        if split == True:
            x_train = self.X[0:args[0]]/255
            yt = self.Y[0:args[0]]
            y_train = self._onehotencoder(yt)
            x_valid = self.X[args[0]:len(self.X)]/255
            yv = self.Y[args[0]:len(self.X)]
            y_valid = self._onehotencoder(yv)
            train_ds = TensorDataset(x_train, y_train) ## for dataloader
            train_dl = DataLoader(train_ds, batch_size=bs, drop_last=False, shuffle=True)
            valid_ds = TensorDataset(x_valid, y_valid)
            valid_dl = DataLoader(valid_ds, batch_size=bs, drop_last=False, shuffle=True)
            return train_dl, valid_dl
        else:
            x_test = self.X/255
            temp_y = self.Y
            y_test = self._onehotencoder(temp_y)
            test_ds = TensorDataset(x_test, y_test)
            test_dl = DataLoader(test_ds, batch_size=bs, drop_last=False, shuffle=False)
            return test_dl
           
class MCLogisticRegression():
    def __init__(self,input_dim,output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Weight and Bias Initializer based on nn.linear in torch
        std = 1/sqrt(input_dim)
        self.weight = torch.zeros(input_dim,output_dim)
        self.weight = nn.init.uniform_(self.weight,-std,std)
        self.bias = torch.zeros(output_dim)
        self.bias = nn.init.uniform_(self.bias,-std,std)
    
    ## Stable softmax
    
    def _softmax(self,input_tensor):
        num = torch.exp(input_tensor - input_tensor.max(dim=1,keepdims=True)[0])
        denom = torch.sum(num,dim=1,keepdim=True)
        out = num/denom
        return out
    
    def lossfunc(self,labels,predictions):
        loss = -torch.sum(labels*torch.log(predictions))
        return loss
    
    def forward(self,input_tensor):
        out = torch.matmul(input_tensor,self.weight) + self.bias
        self.out = self._softmax(out)
    
    def backward(self,input_tensor,labels):
        self.grad_weight = torch.zeros(self.input_dim,self.output_dim)
        self.grad_bias = torch.zeros(self.output_dim)
        self.deviance = (self.out - labels)
        for i in range(self.output_dim):
            self.grad_weight[:,i] = \
            torch.sum(input_tensor.t()*self.deviance[:,i],dim=1)/len(input_tensor)
            self.grad_bias[i] = torch.sum(self.deviance[:,i])/len(input_tensor)
                                 
    def sgd(self,learning_rate):
        self.weight = self.weight - learning_rate*self.grad_weight
        self.bias = self.bias - learning_rate*self.grad_bias
        
    def predict(self,input_tensor):
        out = torch.matmul(input_tensor,self.weight) + self.bias
        out = self._softmax(out)
        return out
    
    def score(self,labels,prediction):
        match = torch.sum(labels.argmax(dim=1) == prediction.argmax(dim=1))
        return match
        
def MNIST(loc,trainfile,testfile,batchsize,learn_rate,splitsize,epochs):   
    train = DataReader()
    train_loader, valid_loader = \
    train.data_processor(loc,trainfile,True,batchsize,splitsize)
    test = DataReader()
    test_loader = test.data_processor(loc,testfile,False,batchsize)
    model = MCLogisticRegression(784,10)

    loss_valid = torch.zeros(epochs,dtype=float)
    accuracy_valid = torch.zeros(epochs,dtype=float)

    for i in range(epochs):
        for j, (X,Y) in enumerate(train_loader):
            model.forward(X)
            model.backward(X,Y)
            model.sgd(learning_rate=learn_rate)
        for k, (VX,VY) in enumerate(valid_loader):    
            outv = model.predict(VX)
            accuracy_valid[i] += model.score(VY,outv)
            loss_valid[i] += model.lossfunc(VY,outv)
        loss_valid[i] =  loss_valid[i]/len(valid_loader.dataset)
        accuracy_valid[i] =  accuracy_valid[i]/len(valid_loader.dataset)

    loss_test = 0.0
    accuracy_test = 0.0

    for k, (TX,TY) in enumerate(test_loader):    
        outt = model.predict(TX)
        accuracy_test += model.score(TY,outt)
        loss_test += model.lossfunc(TY,outt)
        loss_test =  loss_test/len(test_loader.dataset)
    accuracy_test =  accuracy_test/len(test_loader.dataset)
    return loss_valid, accuracy_valid, loss_test, accuracy_test

print("Training Starts")

loc = 'E:/Documents/DLIISC/'
trainfile = 'training.pt'
testfile = 'test.pt'

epochs = 10

counter = 0

learn_rate_lst = list([0.001])
batch_size_lst = list([32])

splitsize = 50000

grid_hyperparam = list(product(learn_rate_lst,batch_size_lst))

valid_loss_store = torch.zeros(len(grid_hyperparam),epochs+2,dtype=float)
valid_acc_store = torch.zeros(len(grid_hyperparam),epochs+2,dtype=float)
test_loss_store = torch.zeros(len(grid_hyperparam),3,dtype=float)
test_acc_store = torch.zeros(len(grid_hyperparam),3,dtype=float)

for lr, batchsize in grid_hyperparam:
    print("Iteration {} Start".format(counter))
    lv, av, lt, at = MNIST(loc,trainfile,testfile,batchsize,lr,splitsize,epochs)
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
        
torch.save(valid_loss_store, loc+'mnist_validation_loss'+str(seed)+'.pt')
torch.save(valid_acc_store, loc+'mnist_validation_accuracy'+str(seed)+'.pt')
torch.save(test_loss_store, loc+'mnist_test_loss'+str(seed)+'.pt')
torch.save(test_acc_store, loc+'mnist_test_accuracy'+str(seed)+'.pt')

fig = plt.figure(figsize=(10,8))
ax = fig.gca(projection='3d')
ax.scatter(xs=test_acc_store[:,1], ys=test_acc_store[:,0], \
                zs=test_acc_store[:,2], cmap = cm.coolwarm, s=30, c=test_acc_store[:,2])
ax.set_xlabel('BatchSize')
ax.set_ylabel('LearningRate')
ax.set_zlabel('Accuracy')
ax.set_title("Test Data Performance")
plt.show()