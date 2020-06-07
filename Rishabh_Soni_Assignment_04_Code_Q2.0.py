""" DNN on MNIST"""

import torch
from numpy import arange
import numpy as np
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
from collections import defaultdict

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
        elif which.lower() == 'tanh':
            return (2/(1+torch.exp(-2*in_tensor))) - 1
        elif which.lower() == 'elu':
            a = 0.2*(torch.exp(in_tensor) - 1)
            val = in_tensor
            mask = a < 0
            val[mask] = a[mask]
            return val
        elif (which.lower() == 'leakyrelu') or (which.lower() == 'lrelu'):
            return torch.max(in_tensor,in_tensor*0.01)
        elif which.lower() == 'gelu':
            tan_input = (in_tensor+0.044715*in_tensor**3)*sqrt(2/np.pi)
            return 0.5*in_tensor*(1+(2/(1+torch.exp(-2*tan_input))) - 1)
        elif which.lower() == 'softplus':
            return torch.log(1+torch.exp(in_tensor))
        elif which.lower() == 'swish':
            return in_tensor*(1/(1+torch.exp(-in_tensor)))
        elif which.lower() == 'selu':
            a = 1.67*(torch.exp(in_tensor) - 1)
            val = in_tensor
            mask = a < 0
            val[mask] = a[mask]
            return val*1.05
        elif which.lower() == 'mish':
            x = torch.log(1+torch.exp(in_tensor))
            return in_tensor*((2/(1+torch.exp(-2*x))) - 1)
        else:
            raise AssertionError("Unexpected activation function!")
    
    def forward_pass(self,input_tensor): 
        self.ylist[0] = input_tensor
        for i in range(1,self.layer_counter+1):
            self.alist[i] = torch.matmul(self.ylist[i-1],self.weightlist[i]) + self.biaslist[i]
            self.ylist[i] = self.activation_apply(self.alist[i],self.layer_activation[i])
        
    def derivative_activation(self,input_tensor,which='relu'):
        if which.lower() == 'relu':
            x = input_tensor.numpy()
            grad = np.where(x > 0,1,0)
            return torch.tensor(grad.astype(np.float32))        
        elif which.lower() == 'tanh':
            return 1 - input_tensor**2
        elif (which.lower() == 'leakyrelu') or (which.lower() == 'lrelu'):
            x = input_tensor.numpy()
            grad = np.where(x > 0,1,0.01)
            return torch.tensor(grad.astype(np.float32))
        elif which.lower() == 'softplus':
            return 1/(1+torch.exp(-input_tensor))
        elif which.lower() == 'swish':
            inter = 1/(1+torch.exp(-input_tensor))
            return input_tensor + inter*(1-input_tensor)
        elif which.lower() == 'gelu':
            main_input = (0.0356774*input_tensor**3 + 0.797885*input_tensor)
            mul_input = 0.0535161*input_tensor**3 + 0.398942*input_tensor
            return 0.5*((2/(1+torch.exp(-2*main_input))) - 1) + \
        mul_input*(4/((torch.exp(main_input) + torch.exp(-1*main_input)))**2) + 0.5
        elif which.lower() == 'elu':
            x = input_tensor.numpy()
            grad = np.where(x > 0,1,x+0.2)
            return torch.tensor(grad.astype(np.float32))
        elif which.lower() == 'selu':
            x = input_tensor.numpy()
            grad = 1.05*np.where(x > 0,1,1.67*np.exp(x))
            return torch.tensor(grad.astype(np.float32))
        elif which.lower() == 'mish':
            omega = \
            torch.exp(3*input_tensor)+4*torch.exp(2*input_tensor)+(6+4*input_tensor)*torch.exp(input_tensor) + \
            4*(1+input_tensor)
            deltasq = ((torch.exp(input_tensor)+1)**2 + 1)**2
            return torch.exp(input_tensor)*torch.div(omega,deltasq)
        elif which.lower() == 'sigmoid':
            x = input_tensor.numpy()
            def func(x):
                return x*(1-x)
            func = np.vectorize(func)
            grad = func(x)
            return torch.tensor(grad.astype(np.float32))
        else:
            raise AssertionError("Unexpected activation function!")
    
    def last_layer_derivative(self,output_tensor,labels):
        return output_tensor - labels
        
    def backprop(self,labels):
        for i in range(self.layer_counter,0,-1):
            if i == self.layer_counter:
                self.grad_alist[i] = self.last_layer_derivative(self.ylist[i],labels)
                self.grad_weightlist[i] = \
                torch.matmul(self.ylist[i-1].t(),self.grad_alist[i])/(len(self.ylist[0])*1.0)
                if self.ifbias[self.layer_counter] == 1:
                    self.grad_biaslist[i] = \
                    torch.sum(self.grad_alist[i],dim=0)/(len(self.ylist[0])*1.0)
                else:
                    self.grad_biaslist[i] = 0
            else:
                self.grad_ylist[i] = torch.matmul(self.grad_alist[i+1],self.weightlist[i+1].t())
                self.grad_alist[i] = \
                self.grad_ylist[i]*self.derivative_activation(self.alist[i],self.layer_activation[i])  
                self.grad_weightlist[i] = \
                torch.matmul(self.ylist[i-1].t(),self.grad_alist[i])/(len(self.ylist[0])*1.0)
                if self.ifbias[self.layer_counter] == 1:
                    self.grad_biaslist[i] = \
                    torch.sum(self.grad_alist[i],dim=0)/(len(self.ylist[0])*1.0)
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
        elif kind.lower() == 'momentum':
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
        elif kind.lower() == 'momentum':
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
            raise AssertionError("Unexpected optimizer!")
    
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
        
def MNIST(location,trainfile,testfile,batchsize,lcnt,node_list,activation_list,\
         bias_ind,optimizer,params,epochs,seed):   
    train = DataReader()
    train_loader, valid_loader = \
    train.data_processor(location,trainfile,True,batchsize,5000)
    test = DataReader()
    test_loader = test.data_processor(location,testfile,False,batchsize)
    
    model = FeedForwardClassifier()
    for i in range(1,lcnt):
        model.add_layer(node_list[i],node_list[i-1],activation_list[i-1],bias_ind[i-1])
    
    model.add_layer(node_list[lcnt],node_list[lcnt-1],activation_list[lcnt-1],bias_ind[lcnt-1])
    
    loss_valid = torch.zeros(epochs,dtype=float)
    accuracy_valid = torch.zeros(epochs,dtype=float)
    
    loss_test = 0.0
    accuracy_test = 0.0

    for i in range(epochs):
        t = 1
        for _, (X,Y) in enumerate(train_loader):
            model.forward_pass(X)
            model.backprop(Y)
            
            if optimizer == 'adam':
                model.optimizer(kind=optimizer,time=t,learning_rate=params[0],beta1=params[1],beta2=params[2])
                t += 1
            elif optimizer == 'momentum':
                model.optimizer(kind=optimizer,time=t,learning_rate=params[0],beta=params[1])
                t += 1
            elif optimizer == 'adagrad':
                model.optimizer(kind=optimizer,time=t,learning_rate=params[0])
                t += 1
            elif optimizer == 'rmsprop':
                model.optimizer(kind=optimizer,time=t,learning_rate=params[0],beta=params[1])
                t += 1
            elif optimizer == 'adamax':
                model.optimizer(kind=optimizer,time=t,learning_rate=params[0],beta1=params[1],beta2=params[2])
                t += 1
            elif optimizer == 'sgd':
                model.optimizer(kind=optimizer,learning_rate=params[0])
                
        for _, (VX,VY) in enumerate(valid_loader):
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
    return loss_valid, accuracy_valid, loss_test, accuracy_test
    
    torch.save(model, location+'mnist_model_seed'+str(seed)+'.pt')

    return loss_valid, accuracy_valid, loss_test, accuracy_test
        
location = 'E:/Documents/DLIISC/'
trainfile = 'training.pt'
testfile = 'test.pt'

epochs = 15

counter = 0

learn_rate_lst = list([0.001, 0.01, 0.05, 0.1])
batch_size_lst = list([1, 32, 128, 1024])

grid_hyperparam = list(product(learn_rate_lst,batch_size_lst))

valid_loss_store = torch.zeros(len(grid_hyperparam),epochs+2,dtype=float)
valid_acc_store = torch.zeros(len(grid_hyperparam),epochs+2,dtype=float)
test_loss_store = torch.zeros(len(grid_hyperparam),3,dtype=float)
test_acc_store = torch.zeros(len(grid_hyperparam),3,dtype=float)

activation_layers = ['relu','relu','relu','softmax']
bias = [True,True,True,True]
optimizer = 'momentum'

# [lr,0.9,0.999]

for lr, batchsize in grid_hyperparam:
    print("Iteration {} Start".format(counter))
    lv, av, lt, at = \
    MNIST(location,trainfile,testfile,batchsize,4,[784,8,8,8,10],activation_layers,\
          bias,optimizer,[lr,0.9],epochs,seed)
    
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
     
torch.save(valid_loss_store, location+'mnist_validation_loss'+str(seed)+'.pt')
torch.save(valid_acc_store, location+'mnist_validation_accuracy'+str(seed)+'.pt')
torch.save(test_loss_store, location+'mnist_test_loss'+str(seed)+'.pt')
torch.save(test_acc_store, location+'mnist_test_accuracy'+str(seed)+'.pt')

fig = plt.figure(figsize=(10,8))
ax = fig.gca(projection='3d')
ax.scatter(xs=test_acc_store[:,1], ys=test_acc_store[:,0], \
                zs=test_acc_store[:,2], cmap = cm.coolwarm, s=30, c=test_acc_store[:,2])
ax.set_xlabel('BatchSize')
ax.set_ylabel('LearningRate')
ax.set_zlabel('Accuracy')
ax.set_title("Test Data Performance")
plt.show()
