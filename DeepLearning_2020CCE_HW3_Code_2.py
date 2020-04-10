
""" 
The code uses already downloaded IRIS data for building a classification model. 
One can implemented as many hidden layers. Relu and Sigmoid are allowed activations only.
Final layer activation can be softmax only.
Optimizer available: ADAM, SGD, NESTEROV SGD, RMSPROP, ADAGRAD, ADAMAX
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
from collections import defaultdict

seed = 42*42 ### Hitchiker guide to galaxy

torch.manual_seed(seed)

class DataReader:
    
    def __init__(self):
        pass
    
    def _read_data(self,loc,filename):
        df = pd.read_csv(loc + filename)
        df = shuffle(df)
        df.reset_index(inplace=True,drop=True)
        return df
        
    def _split_data(self,df,features,target,seed):
        
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
    
    def data_to_tensor(self,loc,filename,bs,seed):
        df = self._read_data(loc,filename)
        features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        train_x, train_y, val_x, val_y, test_x, test_y = self._split_data(df,features,"species",seed)
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
    
class FeedForwardClassifier:
    
    def __init__(self):
        self.layer_counter = 0
        self.input_dim = defaultdict()
        self.output_dim = defaultdict()
        self.weightlist = defaultdict()
        self.layer_activation = defaultdict()
        self.grad_weightlist = defaultdict()
        self.biaslist = defaultdict()
        self.grad_biaslist = defaultdict()
        self.ylist = defaultdict()
        self.alist = defaultdict()
        self.grad_ylist = defaultdict()
        self.grad_alist = defaultdict()
        self.ifbias = defaultdict()
        
    def add_layer(self,output_dim,input_dim,activation,bias=True):
        self.layer_counter += 1
        self.input_dim[self.layer_counter] = input_dim
        self.output_dim[self.layer_counter] = output_dim
        self.weightlist[self.layer_counter] = torch.zeros(input_dim,output_dim)
        torch.manual_seed(seed)
        self.weightlist[self.layer_counter] = \
        nn.init.uniform_(self.weightlist[self.layer_counter],-1/sqrt(input_dim),1/sqrt(input_dim))
        self.layer_activation[self.layer_counter] = activation
        self.grad_weightlist[self.layer_counter] = torch.zeros(input_dim,output_dim)
        if bias == True:
            self.biaslist[self.layer_counter] = torch.zeros(output_dim)
            self.grad_biaslist[self.layer_counter] = torch.zeros(output_dim)
            self.ifbias[self.layer_counter] = 1
        else:
            self.biaslist[self.layer_counter] = 0
            self.ifbias[self.layer_counter] = 0
        
    def _softmax(self,final_tensor):
        num = torch.exp(final_tensor - final_tensor.max(dim=1,keepdims=True)[0])
        denom = torch.sum(num,dim=1,keepdim=True)
        out = num/denom
        return out
    
    def lossfunc(self,labels,predictions):
        loss = -torch.sum(labels*torch.log(predictions))
        return loss
    
    def activation_apply(self,in_tensor,which='relu'):
        if which.lower() == 'relu':
            return torch.max(in_tensor,torch.zeros_like(in_tensor))
        elif which.lower() == 'sigmoid':
            return 1/(1+torch.exp(-in_tensor))
        elif which.lower() == 'softmax':
            return self._softmax(in_tensor)
        else:
            print("Only Relu, Sigmoid and Softmax implemented")
    
    def forward_pass(self,input_tensor):
        self.ylist[0] = input_tensor
        for i in range(1,self.layer_counter+1):
            self.alist[i] = torch.matmul(self.ylist[i-1],self.weightlist[i]) + self.biaslist[i]
            self.ylist[i] = self.activation_apply(self.alist[i],self.layer_activation[i])
        
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
    
    def cross_ent_derivative(self,output_tensor,labels):
        return output_tensor - labels
        
    def backprop(self,labels):
        for i in range(self.layer_counter,0,-1):
            if i == self.layer_counter:
                self.grad_alist[i] = self.cross_ent_derivative(self.ylist[i],labels)
                self.grad_weightlist[i] = \
                torch.matmul(self.ylist[i-1].t(),self.grad_alist[i])/len(self.ylist[0])
                if self.ifbias[self.layer_counter] == 1:
                    self.grad_biaslist[i] = torch.sum(self.grad_alist[i],dim=0)/len(self.ylist[0])
                else:
                    self.grad_biaslist[i] = 0
            else:
                self.grad_ylist[i] = torch.matmul(self.grad_alist[i+1],self.weightlist[i+1].t())
                self.grad_alist[i] = \
                self.grad_ylist[i]*self.derivative_activation(self.alist[i],self.layer_activation[i])                
                self.grad_weightlist[i] = \
                torch.matmul(self.ylist[i-1].t(),self.grad_alist[i])/len(self.ylist[0])
                if self.ifbias[self.layer_counter] == 1:
                    self.grad_biaslist[i] = torch.sum(self.grad_alist[i],dim=0)/len(self.ylist[0])
                else:
                    self.grad_biaslist[i] = 0
                    
    def _init_opt_param(self,kind,t):
        if kind.lower() == 'adam':
            if t == 1:
                self.mw0 = defaultdict()
                self.mw1 = defaultdict()
                self.vw0 = defaultdict()
                self.vw1 = defaultdict()
                self.mhatw1 = defaultdict()
                self.vhatw1 = defaultdict()
                
                self.mb0 = defaultdict()
                self.mb1 = defaultdict()
                self.vb0 = defaultdict()
                self.vb1 = defaultdict()
                self.mhatb1 = defaultdict()
                self.vhatb1 = defaultdict()
                
                for i in range(1,self.layer_counter+1):
                    self.mw0[i] = torch.tensor([0]); self.vw0[i] = torch.tensor([0])
                    self.mb0[i] = torch.tensor([0]); self.vb0[i] = torch.tensor([0])
            else:
                for i in range(1,self.layer_counter+1):
                    self.mw0[i] = self.mw1[i]
                    self.vw0[i] = self.vw1[i]
                    
                    self.mb0[i] = self.mb1[i]
                    self.vb0[i] = self.vb1[i]
        elif kind.lower() == 'nesterov':
            if t == 1:
                self.mw0 = defaultdict()
                self.mw1 = defaultdict()
                self.mb0 = defaultdict()
                self.mb1 = defaultdict()
                
                for i in range(1,self.layer_counter+1):
                    self.mw0[i] = torch.tensor([0]); self.mb0[i] = torch.tensor([0])
            else:
                for i in range(1,self.layer_counter+1):
                    self.mw0[i] = self.mw1[i]
                    
                    self.mb0[i] = self.mb1[i]
        elif kind.lower() == 'adagrad':
            if t == 1:
                self.vw0 = defaultdict()
                self.vw1 = defaultdict()
                self.vb0 = defaultdict()
                self.vb1 = defaultdict()
                
                for i in range(1,self.layer_counter+1):
                    self.vw0[i] = torch.tensor([0]); self.vb0[i] = torch.tensor([0])
            else:
                for i in range(1,self.layer_counter+1):
                    self.vw0[i] = self.vw1[i]
                    
                    self.vb0[i] = self.vb1[i]
        elif kind.lower() == 'rmsprop':
            if t == 1:
                self.vw0 = defaultdict()
                self.vw1 = defaultdict()
                self.vb0 = defaultdict()
                self.vb1 = defaultdict()
                
                for i in range(1,self.layer_counter+1):
                    self.vw0[i] = torch.tensor([0]); self.vb0[i] = torch.tensor([0])
            else:
                for i in range(1,self.layer_counter+1):
                    self.vw0[i] = self.vw1[i]
                    
                    self.vb0[i] = self.vb1[i]
        elif kind.lower() == 'adamax':
            if t == 1:
                self.mw0 = defaultdict()
                self.mw1 = defaultdict()
                self.vw0 = defaultdict()
                self.vw1 = defaultdict()
                self.mhatw1 = defaultdict()
                
                self.mb0 = defaultdict()
                self.mb1 = defaultdict()
                self.vb0 = defaultdict()
                self.vb1 = defaultdict()
                self.mhatb1 = defaultdict()
                
                for i in range(1,self.layer_counter+1):
                    self.mw0[i] = torch.tensor([0]); self.vw0[i] = torch.tensor([0])
                    self.mb0[i] = torch.tensor([0]); self.vb0[i] = torch.tensor([0])
            else:
                for i in range(1,self.layer_counter+1):
                    self.mw0[i] = self.mw1[i]
                    self.vw0[i] = self.vw1[i]
                    
                    self.mb0[i] = self.mb1[i]
                    self.vb0[i] = self.vb1[i]
                
    def optimizer(self,kind,**kwargs):
        if kind.lower() == 'sgd':
            learning_rate = kwargs['learning_rate']
            for i in range(1,self.layer_counter+1):
                self.weightlist[i] = self.weightlist[i] - learning_rate*self.grad_weightlist[i]
                self.biaslist[i] = self.biaslist[i] - learning_rate*self.grad_biaslist[i]
        elif kind.lower() == 'adam':
            t = kwargs['time']
            learning_rate = kwargs['learning_rate']
            beta1 = kwargs['beta1']
            beta2 = kwargs['beta2']
            self._init_opt_param(kind,t)             
            for i in range(1,self.layer_counter+1):
                self.mw1[i] = beta1*self.mw0[i] + (1-beta1)*self.grad_weightlist[i]
                self.vw1[i] = beta2*self.vw0[i] + (1-beta2)*(self.grad_weightlist[i])**2
                self.mhatw1[i] = self.mw1[i]/(1-beta1**t)
                self.vhatw1[i] = self.vw1[i]/(1-beta2**t)
                
                self.mb1[i] = beta1*self.mb0[i] + (1-beta1)*self.grad_biaslist[i]
                self.vb1[i] = beta2*self.vb0[i] + (1-beta2)*(self.grad_biaslist[i])**2
                self.mhatb1[i] = self.mb1[i]/(1-beta1**t)
                self.vhatb1[i] = self.vb1[i]/(1-beta2**t)
                
                self.weightlist[i] = \
                self.weightlist[i] - learning_rate*torch.div(self.mhatw1[i],(torch.sqrt(self.vhatw1[i])+1e-8))
                
                self.biaslist[i] = \
                self.biaslist[i] - learning_rate*torch.div(self.mhatb1[i],(torch.sqrt(self.vhatb1[i])+1e-8))
        elif kind.lower() == 'nesterov':
            t = kwargs['time']
            learning_rate = kwargs['learning_rate']
            beta = kwargs['beta']
            self._init_opt_param(kind,t)             
            for i in range(1,self.layer_counter+1):
                self.mw1[i] = beta*self.mw0[i] + (1-beta)*self.grad_weightlist[i]
                self.mb1[i] = beta*self.mb0[i] + (1-beta)*self.grad_biaslist[i]
                self.weightlist[i] = self.weightlist[i] - learning_rate*self.mw1[i]
                self.biaslist[i] = self.biaslist[i] - learning_rate*self.mb1[i]
        elif kind.lower() == 'adagrad':
            t = kwargs['time']
            learning_rate = kwargs['learning_rate']
            self._init_opt_param(kind,t)  
            for i in range(1,self.layer_counter+1):
                self.vw1[i] = self.vw0[i] + self.grad_weightlist[i]**2
                self.vb1[i] = self.vb0[i] + self.grad_biaslist[i]**2
                self.weightlist[i] = \
                self.weightlist[i]-learning_rate*torch.div(self.grad_weightlist[i],torch.sqrt(self.vw1[i]+1e-8))
                self.biaslist[i] = \
                self.biaslist[i]-learning_rate*torch.div(self.grad_biaslist[i],torch.sqrt(self.vb1[i]+1e-8))
        elif kind.lower() == 'rmsprop':
            t = kwargs['time']
            learning_rate = kwargs['learning_rate']
            beta = kwargs['beta']
            self._init_opt_param(kind,t)  
            for i in range(1,self.layer_counter+1):
                self.vw1[i] = beta*self.vw0[i] + (1-beta)*self.grad_weightlist[i]**2
                self.vb1[i] = beta*self.vb0[i] + (1-beta)*self.grad_biaslist[i]**2
                self.weightlist[i] = \
                self.weightlist[i]-learning_rate*torch.div(self.grad_weightlist[i],torch.sqrt(self.vw1[i]+1e-8))
                self.biaslist[i] = \
                self.biaslist[i]-learning_rate*torch.div(self.grad_biaslist[i],torch.sqrt(self.vb1[i]+1e-8))
        elif kind.lower() == 'adamax':
            t = kwargs['time']
            learning_rate = kwargs['learning_rate']
            beta1 = kwargs['beta1']
            beta2 = kwargs['beta2']
            self._init_opt_param(kind,t)             
            for i in range(1,self.layer_counter+1):
                self.mw1[i] = beta1*self.mw0[i] + (1-beta1)*self.grad_weightlist[i]
                self.vw1[i] = torch.max(beta2*self.vw0[i],torch.abs(self.grad_weightlist[i]))
                self.mhatw1[i] = self.mw1[i]/(1-beta1**t)
                
                self.mb1[i] = beta1*self.mb0[i] + (1-beta1)*self.grad_biaslist[i]
                self.vb1[i] = torch.max(beta2*self.vb0[i],torch.abs(self.grad_biaslist[i]))
                self.mhatb1[i] = self.mb1[i]/(1-beta1**t)
                
                self.weightlist[i] = \
                self.weightlist[i] - learning_rate*torch.div(self.mhatw1[i],(self.vw1[i]+1e-8))
                
                self.biaslist[i] = \
                self.biaslist[i] - learning_rate*torch.div(self.mhatb1[i],(self.vb1[i]+1e-8))
        else:
            print("Only SGD, ADAM, NESTEROV, ADAGRAD, RMSPROP & ADAMAX implemented")
    
    def predict(self,input_tensor):
        ylist = defaultdict()
        alist = defaultdict()
        ylist[0] = input_tensor
        for i in range(1,self.layer_counter+1):
            alist[i] = torch.matmul(ylist[i-1],self.weightlist[i]) + self.biaslist[i]
            ylist[i] = self.activation_apply(alist[i],self.layer_activation[i])
        return ylist[self.layer_counter]
        
    def score(self,labels,prediction):
        ## Every row is the output for each data point hence argmax on dimension 1
        match = torch.sum(labels.argmax(dim=1) == prediction.argmax(dim=1))
        return match
        
def IRIS(location,filename,batchsize,lcnt,node_list,activation_list,\
         bias_ind,optimizer,params,epochs,seed):   
    data_model = DataReader()
    train_loader, valid_loader, test_loader = \
    data_model.data_to_tensor(location,filename,batchsize,seed)
    model = FeedForwardClassifier()
    for i in range(1,lcnt):
        model.add_layer(node_list[i],node_list[i-1],activation_list[i-1],bias_ind[i-1])
    
    model.add_layer(node_list[lcnt],node_list[lcnt-1],activation_list[lcnt-1],bias_ind[lcnt-1])

    loss_train = torch.zeros(epochs,dtype=float)
    accuracy_train = torch.zeros(epochs,dtype=float)
    
    loss_valid = torch.zeros(epochs,dtype=float)
    accuracy_valid = torch.zeros(epochs,dtype=float)
    
    loss_test = torch.zeros(epochs,dtype=float)
    accuracy_test = torch.zeros(epochs,dtype=float)

    for i in range(epochs):
        t = 1
        for _, (X,Y) in enumerate(train_loader):
            model.forward_pass(X)
            model.backprop(Y)
            
            if optimizer == 'adam':
                model.optimizer(kind=optimizer,time=t,learning_rate=params[0],beta1=params[1],beta2=params[2])
                t += 1
            if optimizer == 'nesterov':
                model.optimizer(kind=optimizer,time=t,learning_rate=params[0],beta=params[1])
                t += 1
            if optimizer == 'adagrad':
                model.optimizer(kind=optimizer,time=t,learning_rate=params[0])
                t += 1
            if optimizer == 'rmsprop':
                model.optimizer(kind=optimizer,time=t,learning_rate=params[0],beta=params[1])
                t += 1
            if optimizer == 'adamax':
                model.optimizer(kind=optimizer,time=t,learning_rate=params[0],beta1=params[1],beta2=params[2])
                t += 1
            if optimizer == 'sgd':
                model.optimizer(kind=optimizer,learning_rate=params[0])
                
        for _, (X,Y) in enumerate(train_loader): 
            outr = model.predict(X)
            accuracy_train[i] += model.score(Y,outr)
            loss_train[i] += model.lossfunc(Y,outr)
        loss_train[i] =  loss_train[i]/len(train_loader.dataset)
        accuracy_train[i] =  accuracy_train[i]/len(train_loader.dataset)
        for _, (VX,VY) in enumerate(valid_loader):
            outv = model.predict(VX)
            accuracy_valid[i] += model.score(VY,outv)
            loss_valid[i] += model.lossfunc(VY,outv)
        loss_valid[i] =  loss_valid[i]/len(valid_loader.dataset)
        accuracy_valid[i] =  accuracy_valid[i]/len(valid_loader.dataset)
        for _, (TX,TY) in enumerate(test_loader):
            outt = model.predict(TX)
            accuracy_test[i] += model.score(TY,outt)
            loss_test[i] += model.lossfunc(TY,outt)
        loss_test[i] =  loss_test[i]/len(test_loader.dataset)
        accuracy_test[i] =  accuracy_test[i]/len(test_loader.dataset)
    
    torch.save(model, location+'iris_model_seed'+str(seed)+'.pt')

    return model,loss_train, accuracy_train, loss_valid, accuracy_valid, loss_test, accuracy_test
        
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
    model, ltr, atr, lv, av, lt, at = \
    IRIS(location,filename,batchsize,3,[4,10,10,3],['relu','relu','softmax'],[True,True,True],\
         'adamax',[lr,0.9,0.999],epochs,seed)
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

