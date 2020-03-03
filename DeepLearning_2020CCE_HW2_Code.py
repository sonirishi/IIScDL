# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 23:26:41 2020

@author: Admin
"""

import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

torch.manual_seed(1988)

## Dataset is abstract class

class read_data:
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
    
    def _onehot(self,tensor):
        y_train = torch.zeros((len(tensor),len(tensor.unique())))
        y_train[np.arange(len(y_train)),tensor] = 1
        return y_train
    
    def preparedata(self,location,filename,split,bs,*args):
        self._load_data(location,filename)
        self.X = torch.reshape(self.X,(self.X.size()[0],self.X.size()[1]*self.X.size()[2]))
        if split == True:
            x_train = self.X[0:args[0]]/255
            yt = self.Y[0:args[0]]
            y_train = self._onehot(yt)
            x_valid = self.X[args[0]:len(self.X)]/255
            yv = self.Y[args[0]:len(self.X)]
            y_valid = self._onehot(yv)
            train_ds = TensorDataset(x_train, y_train) ## for dataloader
            train_dl = DataLoader(train_ds, batch_size=bs, drop_last=False, shuffle=True)
            valid_ds = TensorDataset(x_valid, y_valid)
            valid_dl = DataLoader(valid_ds, batch_size=bs, drop_last=False, shuffle=True)
            return train_dl, valid_dl
        else:
            x_test = self.X/255
            temp_y = self.Y
            y_test = self._onehot(temp_y)
            test_ds = TensorDataset(x_test, y_test)
            test_dl = DataLoader(test_ds, batch_size=bs, drop_last=False, shuffle=False)
            return test_dl
           
class LogisticRegression():
    def __init__(self,input_dim,output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        std = 1/sqrt(input_dim)
        self.weight = torch.zeros(input_dim,output_dim)
        self.weight = nn.init.uniform_(self.weight,-std,std)
        self.bias = torch.zeros(output_dim)
        self.bias = nn.init.uniform_(self.bias,-std,std)
        
    def _softmax(self,input_tensor):
        out = torch.exp(input_tensor)/torch.sum(torch.exp(input_tensor),dim=1,keepdim=True)
        return out
    
    def lossfunc(self,output_tensor,predicted):
        loss = -torch.sum(output_tensor*torch.log(predicted))
        return loss
    
    def forward(self,input_tensor):
        out = torch.matmul(input_tensor,self.weight) + self.bias
        self.out = self._softmax(out)
        return self.out
    
    def backward(self,input_tensor,output_tensor):
        self.grad_weight = torch.zeros(self.input_dim,self.output_dim)
        self.grad_bias = torch.zeros(self.output_dim)
        self.deviance = (self.out - output_tensor)
        for i in range(self.output_dim):
            self.grad_weight[:,i] = \
            torch.sum(input_tensor.t()*self.deviance[:,i],dim=1)/len(input_tensor)
            self.grad_bias[i] = torch.sum(self.deviance[:,i])/len(input_tensor)
            
#    def backward(self,input_tensor,output_tensor):
#        self.grad_weight = torch.zeros(self.input_dim,self.output_dim)
#        self.grad_bias = torch.zeros(self.output_dim)
#        self.deviance = (self.out - output_tensor)
#        for i in range(self.output_dim):
#            for j in range(len(input_tensor)):
#                self.grad_weight[:,i] += self.deviance[j,i]*input_tensor[j,]
#                self.grad_bias[i] += self.deviance[j,i]
#        self.grad_weight = self.grad_weight/len(input_tensor)
#        self.grad_bias = self.grad_bias/len(input_tensor)
                     
    def sgd(self,learning_rate):
        self.weight = self.weight - learning_rate*self.grad_weight
        self.bias = self.bias - learning_rate*self.grad_bias
        
    def predict(self,input_tensor):
        out = torch.matmul(input_tensor,self.weight) + self.bias
        out = self._softmax(out)
        return out
        
    
train = read_data()

batchsize = 32

train_loader, valid_loader = train.preparedata('E:/Documents/DLIISC/','training.pt',\
                                               True,batchsize,50000)

test = read_data()

test_loader = test.preparedata('E:/Documents/DLIISC/','test.pt',False,batchsize)

model = LogisticRegression(784,10)

epochs = 10

# =============================================================================
# p = iter(train_loader)
# X,Y = next(p)
# 
# output = model.forward(X)
# 
# k = Y - output
# 
# model.backward(X,Y)
# 
# model.sgd(0.001)
# 
# j = X.float().t()*k[:,0]
# =============================================================================

learn_rate = 0.001

loss_valid = torch.zeros(epochs)

for i in range(epochs):
    for j, (X,Y) in enumerate(train_loader):
        output = model.forward(X)
        model.backward(X,Y)
        model.sgd(learning_rate=learn_rate)
    for k, (VX,VY) in enumerate(valid_loader):    
        outv = model.predict(VX)
        loss_valid[i] += model.lossfunc(VY,outv)
    loss_valid[i] =  loss_valid[i]/len(valid_loader.dataset)

loss_test = 0

for k, (TX,TY) in enumerate(test_loader):    
    outt = model.predict(TX)
    loss_test += model.lossfunc(TY,outt)
loss_test =  loss_test/len(test_loader.dataset)
            
        

    
