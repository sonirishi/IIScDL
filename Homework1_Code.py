# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 18:26:36 2020

@author: Admin
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from numpy.random import multivariate_normal
from numpy.linalg import qr, norm, eigh
from numpy import identity, transpose, matmul, triu, diag, zeros, absolute, cov, \
column_stack, mean, std, sqrt
from sklearn.decomposition import PCA

print(pd.__version__)
print(sns.__version__)

### Q2 Implementation of scaler and whitening

class normalise:
    def __init__(self,technique = 'standardise'):
        self.technique = technique.lower()
    
    def fit(self,matrix):
        if self.technique == 'standardise':
            self.mean = mean(matrix,axis=0)
            self.std = std(matrix,axis=0)
            self.output = (matrix - self.mean)/self.std
            return self.output, self.mean, self.std
        elif self.technique == 'whiten':
            self.mean = mean(matrix,axis=0)
            self.std = std(matrix,axis=0)            
            self.data_cov = cov(matrix,rowvar=False)
            self.evals, self.evecs = eigh(self.data_cov)
            demean_random = matrix - self.mean
            self.whiten_random = matmul(demean_random,self.evecs.T)/sqrt(self.evals)
            return self.whiten_random, self.data_cov
        else:
            print('Technique not implemented')

avg = [3, 4]
covr = [[2, 1.5], [1.5, 2.5]]  # diagonal covariance
random_data = multivariate_normal(avg, covr, 100)

scl_ri = normalise(technique='standardise')
scale_random, datamean, datastd = scl_ri.fit(random_data)

sns.scatterplot(x=scale_random[:,0],y=scale_random[:,1])
plt.xlabel("X")
plt.ylabel("Y")
plt.title('Scaled Data')

### Scale using Scikit Learn ###

scl = StandardScaler()

scale_data_sklean = scl.fit_transform(random_data)

sns.scatterplot(x=scale_data_sklean[:,0],y=scale_data_sklean[:,1])
plt.xlabel("X")
plt.ylabel("Y")
plt.title('Scaled Data')

## Covariance matrix and eigendecomposition #####

whiten_ri = normalise(technique='whiten')
whiten_random, cov_matrix = whiten_ri.fit(random_data)

sns.scatterplot(x=whiten_random[:,0],y=whiten_random[:,1])
plt.xlabel("X")
plt.ylabel("Y")
plt.title('Whiten Data')

## Compare with PCA O/P ##

pca = PCA(n_components=2)
scale_random_pca = pca.fit_transform(scale_random)

sns.scatterplot(x=scale_random_pca[:,0],y=scale_random_pca[:,1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title('PrinComp')

### Q3 PCA implementation from scratch ###

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# load dataset into Pandas DataFrame
df = pd.read_csv(url, names=['sepal length','sepal width','petal length',\
                             'petal width','target'])

features = ['sepal length', 'sepal width', 'petal length', 'petal width']
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['target']].values

class svd_decompose:
    def __init__(self,tol,iterations):
        self.tolerance = tol
        self.max_iter = iterations
    
    def fit(self,matrix):
        self.U = identity(matrix.shape[0])
        self.V = identity(matrix.shape[1])
        self.S = transpose(matrix)
        self.error = 1000000
        counter = 0
        while self.error >= self.tolerance and counter <= self.max_iter:
            q1,r1 = qr(transpose(self.S))
            self.U = matmul(self.U,q1)
            self.S = r1
            q2,r2 = qr(transpose(self.S))
            self.V = matmul(self.V,q2)
            self.S = r2
            E = norm(triu(self.S,1))
            F = norm(diag(self.S))
            if F == 0:
                F = 1
            self.error = E/F
            counter += 1
            #print(counter)
        SS = diag(self.S)
        self.S = zeros(matrix.shape)
        for i in range(len(SS)):
            self.S[i,i] = abs(SS[i])
            if SS[i] < 0:
                self.U[:,i]=-self.U[:,i]
        return self.U, diag(self.S), self.V
    
svd_ri = svd_decompose(0.00000000001,100)

scale = StandardScaler()

X = scale.fit_transform(x)

U,S,V = svd_ri.fit(X)

newx1 = U[:,0]*S[0]
newx2 = U[:,1]*S[1]

X_pca = column_stack((newx1,newx2))

pca_data = column_stack((X_pca,y))

sns.scatterplot(x=pca_data[:,0],y=pca_data[:,1],hue=pca_data[:,2])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title('Self Implemeted PCA Features with Class')

### Test using actual PCA implementation in scikit-learn

pca = PCA(n_components=2)
X_pca_sklearn = pca.fit_transform(X)

pca_data_sklearn = column_stack((X_pca_sklearn,y))

sns.scatterplot(x=pca_data_sklearn[:,0],y=pca_data_sklearn[:,1],hue=pca_data_sklearn[:,2])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title('PCA Features with Class')