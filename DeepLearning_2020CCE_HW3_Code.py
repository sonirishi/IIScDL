
""" 
Commented out version of the code is to
download the IRIS data. 
The uncommented data uses already downloaded IRIS data
for building a logistic model with 2 hidden layers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch
from math import sqrt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from itertools import product
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 

seed = 42 ### Hitchiker guide to galaxy

torch.manual_seed(seed)

class DataReader:
    
    def __init__(self):
        pass
    
    def _read_data(self,loc,filename):
        df = pd.read_csv(loc + filename)
        df = shuffle(df)
        df.reset_index(inplace=True,drop=True)
        return df
        
    def _split_data(self,df,features,target):
        
        df_x = df.loc[:, features].values
        df_y = df.loc[:,[target]].values
        
        df_y = LabelEncoder().fit_transform(np.ravel(df_y))
        
        train_x, test_x, train_y, test_y = \
        train_test_split(df_x,df_y,test_size=0.33,random_state=seed,stratify=df_y)

        train_x, val_x, train_y, val_y = \
        train_test_split(train_x,train_y,test_size=0.2,random_state=seed,stratify=train_y)
        
        return train_x, train_y, val_x, val_y, test_x, test_y
        
    def _scale_data(self,train_x,val_x,test_x):
        clf = StandardScaler()
        train_x_scale = clf.fit_transform(train_x)
        val_x_scale = clf.transform(val_x)
        test_x_scale = clf.transform(test_x)
        return train_x_scale, val_x_scale, test_x_scale
    
    def _one_hot_encoder(self,arr):
        arr_1 = np.zeros((arr.shape[0],len(np.unique(arr))))
        arr_1[np.arange(arr.shape[0]),arr] = 1
        return arr_1
    
    def data_to_tensor(self,loc,filename,bs):
        df = self._read_data(loc,filename)
        features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        train_x, train_y, val_x, val_y, test_x, test_y = self._split_data(df,features,"species")
        train_x_scale, val_x_scale, test_x_scale = self._scale_data(train_x,val_x,test_x)
        train_y_oh = self._one_hot_encoder(train_y)
        val_y_oh = self._one_hot_encoder(val_y)
        test_y_oh = self._one_hot_encoder(test_y)
        train_x_scale = torch.Tensor(train_x_scale)
        train_y_oh = torch.Tensor(train_y_oh)
        val_x_scale = torch.Tensor(val_x_scale)
        val_y_oh = torch.Tensor(val_y_oh)
        test_x_scale = torch.Tensor(test_x_scale)
        test_y_oh = torch.Tensor(test_y_oh)
        
        train_ds = TensorDataset(train_x_scale, train_y_oh) ## for dataloader
        train_dl = DataLoader(train_ds, batch_size=bs, drop_last=False, shuffle=True)
        
        val_ds = TensorDataset(val_x_scale, val_y_oh) ## for dataloader
        val_dl = DataLoader(val_ds, batch_size=val_x_scale.shape[0], drop_last=False, shuffle=False)
        
        test_ds = TensorDataset(test_x_scale, test_y_oh) ## for dataloader
        test_dl = DataLoader(test_ds, batch_size=test_x_scale.shape[0], drop_last=False, shuffle=False)
        
        return train_dl, val_dl, test_dl
    
class TwoHiddenDL:
    
    def __init__(self, input_dim, hidden1, hidden2, output_dim):
        self.input_dim = input_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.output_dim = output_dim
        self.weight_l1 = torch.zeros(self.input_dim,self.hidden1)
        self.weight_l1 = nn.init.uniform_(self.weight_l1,-1/sqrt(self.input_dim),1/sqrt(self.input_dim))
        self.bias1 = torch.zeros(self.hidden1)
        self.weight_l2 = torch.zeros(self.hidden1,self.hidden2)
        self.weight_l2 = nn.init.uniform_(self.weight_l2,-1/sqrt(self.hidden1),1/sqrt(self.hidden1))
        self.bias2 = torch.zeros(self.hidden2)
        self.weight_l3 = torch.zeros(self.hidden2,self.output_dim)
        self.weight_l3 = nn.init.uniform_(self.weight_l3,-1/sqrt(self.hidden2),1/sqrt(self.hidden2))
        self.bias3 = torch.zeros(self.output_dim)
        
    def _softmax(self,final_tensor):
        num = torch.exp(final_tensor - final_tensor.max(dim=1,keepdims=True)[0])
        denom = torch.sum(num,dim=1,keepdim=True)
        out = num/denom
        return out
    
    def lossfunc(self,labels,predictions):
        loss = -torch.sum(labels*torch.log(predictions))
        return loss
    
    def activation(self,in_tensor,which='relu'):
        if which.lower() == 'relu':
            return torch.max(in_tensor,torch.zeros_like(in_tensor))
        elif which.lower() == 'sigmoid':
            return 1/(1+torch.exp(-in_tensor))
        else:
            print("Only Relu and Sigmoid implemented")
    
    def forward_pass(self,input_tensor):
        self.a1 = torch.matmul(input_tensor,self.weight_l1) + self.bias1
        self.y1 = self.activation(self.a1,'relu')
        self.a2 = torch.matmul(self.y1,self.weight_l2) + self.bias2
        self.y2 = self.activation(self.a2,'relu')
        self.a3 = torch.matmul(self.y2,self.weight_l3) + self.bias3
        self.y3 = self._softmax(self.a3)
        
    def derivative_activation(self,input_tensor,which='relu'):
        if which.lower() == 'relu':
            x = input_tensor.numpy()
            grad = np.where(x > 0,1,0)
            return torch.tensor(grad)
        elif which.lower() == 'sigmoid':
            x = input_tensor.numpy()
            def func(x):
                return x*(1-x)
            func = np.vectorize(func)
            grad = func(x)
            return torch.tensor(grad)
        else:
            print("Only Relu and Sigmoid implemented")
        
    def backprop(self,input_tensor,labels):
        self.grad_weight1 = torch.zeros(self.input_dim,self.hidden1)
        self.grad_bias1 = torch.zeros(self.hidden1)
        self.grad_weight2 = torch.zeros(self.hidden1,self.hidden2)
        self.grad_bias2 = torch.zeros(self.hidden2)
        self.grad_weight3 = torch.zeros(self.hidden2,self.output_dim)
        self.grad_bias3 = torch.zeros(self.output_dim)
        self.grad_a3 = self.y3 - labels
        self.grad_y2 = torch.matmul(self.grad_a3,self.weight_l3.t())
        self.grad_weight3 = torch.matmul(self.y2.t(),self.grad_a3)/len(input_tensor)
        self.grad_bias3 = torch.sum(self.grad_a3)/len(input_tensor)
        self.grad_a2 = self.grad_y2*self.derivative_activation(self.a2,'relu')
        self.grad_y1 = torch.matmul(self.grad_a2,self.weight_l2.t())
        self.grad_weight2 = torch.matmul(self.y1.t(),self.grad_a2)/len(input_tensor)
        self.grad_bias2 = torch.sum(self.grad_a2)/len(input_tensor)
        self.grad_a1 = self.grad_y1*self.derivative_activation(self.a1,'relu')
        self.grad_weight1 = torch.matmul(input_tensor.t(),self.grad_a1)/len(input_tensor)
        self.grad_bias1 = torch.sum(self.grad_a1)/len(input_tensor)
        
    def sgd(self,learning_rate):
        self.weight_l1 = self.weight_l1 - learning_rate*self.grad_weight1
        self.weight_l2 = self.weight_l2 - learning_rate*self.grad_weight2
        self.weight_l3 = self.weight_l3 - learning_rate*self.grad_weight3
        self.bias1 = self.bias1 - learning_rate*self.grad_bias1
        self.bias2 = self.bias2 - learning_rate*self.grad_bias2
        self.bias3 = self.bias3 - learning_rate*self.grad_bias3
    
    def predict(self,input_tensor):
        a1 = torch.matmul(input_tensor,self.weight_l1) + self.bias1
        y1 = self.activation(a1,'relu')
        a2 = torch.matmul(y1,self.weight_l2) + self.bias2
        y2 = self.activation(a2,'relu')
        a3 = torch.matmul(y2,self.weight_l3) + self.bias3
        y3 = self._softmax(a3)
        return y3
        
    def score(self,labels,prediction):
        ## Every row is the output for each data point hence argmax on dimension 1
        match = torch.sum(labels.argmax(dim=1) == prediction.argmax(dim=1))
        return match
        
def IRIS(location,filename,batchsize,learn_rate,epochs):   
    data_model = DataReader()
    train_loader, valid_loader, test_loader = \
    data_model.data_to_tensor(location,filename,batchsize)
    model = TwoHiddenDL(4,10,10,3)

    loss_train = torch.zeros(epochs,dtype=float)
    accuracy_train = torch.zeros(epochs,dtype=float)
    
    loss_valid = torch.zeros(epochs,dtype=float)
    accuracy_valid = torch.zeros(epochs,dtype=float)
    
    loss_test = torch.zeros(epochs,dtype=float)
    accuracy_test = torch.zeros(epochs,dtype=float)

    for i in range(epochs):
        for j, (X,Y) in enumerate(train_loader):
            model.forward_pass(X)
            model.backprop(X,Y)
            model.sgd(learning_rate=learn_rate)
        for j, (X,Y) in enumerate(train_loader): 
            outr = model.predict(X)
            accuracy_train[i] += model.score(Y,outr)
            loss_train[i] += model.lossfunc(Y,outr)
        loss_train[i] =  loss_train[i]/len(train_loader.dataset)
        accuracy_train[i] =  accuracy_train[i]/len(train_loader.dataset)
        for k, (VX,VY) in enumerate(valid_loader):
            outv = model.predict(VX)
            accuracy_valid[i] += model.score(VY,outv)
            loss_valid[i] += model.lossfunc(VY,outv)
        loss_valid[i] =  loss_valid[i]/len(valid_loader.dataset)
        accuracy_valid[i] =  accuracy_valid[i]/len(valid_loader.dataset)
        for k, (TX,TY) in enumerate(test_loader):
            outt = model.predict(TX)
            accuracy_test += model.score(TY,outt)
            loss_test += model.lossfunc(TY,outt)
        loss_test =  loss_test/len(test_loader.dataset)
        accuracy_test =  accuracy_test/len(test_loader.dataset)

    return loss_train, accuracy_train, loss_valid, accuracy_valid, loss_test, accuracy_test
        
filename = 'iris.csv'

location = 'E:/Documents/DLIISC/'

epochs = 20

counter = 0

learn_rate_lst = list([0.001, 0.01, 0.05])
batch_size_lst = list([4,8,16])

grid_hyperparam = list(product(learn_rate_lst,batch_size_lst))

train_loss_store = torch.zeros(len(grid_hyperparam),epochs+2,dtype=float)
train_acc_store = torch.zeros(len(grid_hyperparam),epochs+2,dtype=float)
valid_loss_store = torch.zeros(len(grid_hyperparam),epochs+2,dtype=float)
valid_acc_store = torch.zeros(len(grid_hyperparam),epochs+2,dtype=float)
test_loss_store = torch.zeros(len(grid_hyperparam),epochs+2,dtype=float)
test_acc_store = torch.zeros(len(grid_hyperparam),epochs+2,dtype=float)

for lr, batchsize in grid_hyperparam:
    print("Iteration {} Start".format(counter))
    ltr, atr, lv, av, lt, at = IRIS(location,filename,batchsize,lr,epochs)
    ## Store train accuracy and loss after each epoch
    train_loss_store[counter,0] = lr
    train_loss_store[counter,1] = batchsize
    train_acc_store[counter,0] = lr
    train_acc_store[counter,1] = batchsize
    train_loss_store[counter,2:] = ltr
    train_acc_store[counter,2:] = atr
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
    test_loss_store[counter,2:] = lt
    test_acc_store[counter,2:] = at
    
    print("Iteration {} Done".format(counter))
                
    counter += 1    

torch.save(train_loss_store, location+'iris_train_loss'+str(seed)+'.pt')
torch.save(train_acc_store, location+'iris_train_accuracy'+str(seed)+'.pt')        
torch.save(valid_loss_store, location+'iris_validation_loss'+str(seed)+'.pt')
torch.save(valid_acc_store, location+'iris_validation_accuracy'+str(seed)+'.pt')
torch.save(test_loss_store, location+'iris_test_loss'+str(seed)+'.pt')
torch.save(test_acc_store, location+'iris_test_accuracy'+str(seed)+'.pt')
