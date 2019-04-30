
# coding: utf-8

# In[124]:


import struct
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# In[125]:


import os
# lead training data
def load_mnist():
    """Load MNIST data"""
    labels_path = os.path.join('train-labels-idx1-ubyte')
    images_path = os.path.join('train-images-idx3-ubyte')
    
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype=np.uint8)
        
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)# each row represents a image

    return images, labels

X_train, y_train = load_mnist()
X_train = X_train.astype(float)
y_train = y_train.astype(float)


# In[126]:


X_train.shape


# In[127]:


X_train[0,1:200]


# In[128]:


y_train.shape


# In[129]:


y_train[0:10]


# In[130]:


# load testing data
def load_mnist():
    """Load MNIST data"""
    labels_path = os.path.join('t10k-labels-idx1-ubyte')
    images_path = os.path.join('t10k-images-idx3-ubyte')
    
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype=np.uint8)
        
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)# each row represents a image

    return images, labels

X_test, y_test = load_mnist()
X_train = X_train.astype(float)
y_train = y_train.astype(float)
X_test.shape


# In[131]:


# export image with label 0 and label 9
X_train = X_train[(y_train == 0) | (y_train == 9)]
y_train = y_train[(y_train == 0) | (y_train == 9)]
y = y_train.astype(float)
y_train.shape


# In[132]:


y_train[0:10]


# In[133]:


X_test = X_test[(y_test == 0) | (y_test == 9)]
y_test = y_test[(y_test == 0) | (y_test == 9)]
y_test = y_test.astype(float)


# In[134]:


y_test[0:10]


# In[135]:


for i,k in enumerate(y_train):
    if k == 9:
        y_train[i]= 1
    else:
        y_train[i]= -1


# In[136]:


y_train[0:10]


# In[137]:


for i,k in enumerate(y_test):
    if k == 9:
        y_test[i]= 1
    else:
        y_test[i]= -1


# In[138]:


y_test[0:10]


# In[139]:


import math


# In[140]:


## generate gaussian matrix
def gen_gaussion(m):
    k = m*n
    s = np.random.normal(0, 1, k).reshape(m, n)
    Gaussion = np.mat(s)
    return Gaussion

n = len(X_train[1,])
m = int(n * math.log(n))
Gau = gen_gaussion(m)


# In[141]:


# mapping X into a higher dimension
def mapping(X):
    X_matrix = np.transpose([np.mat(X)])
    X_map = np.transpose(1/math.sqrt(m) * np.sign(Gau * X_matrix))
    return X_map 

X_maptrain = mapping(X_train)
X_maptest = mapping(X_test)


# In[142]:


X_maptrain.shape


# In[170]:


H


# In[166]:


# Generate matrix H for cvxopt
X = X_maptrain
y = y_train.reshape(-1, 1) * 1.
i,j = X.shape
X_dash = np.zeros(shape=(i,j))
for t,item_y in enumerate(y):
        X_dash[t,] = item_y * X[t,]
        
H = np.dot(X_dash , X_dash.T) * 1.


# In[171]:


from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers


# Converting into cvxopt format
P = cvxopt_matrix(H)
q = cvxopt_matrix(-np.ones((i, 1)))
G = cvxopt_matrix(-np.eye(i))
h = cvxopt_matrix(np.zeros(i))
A = cvxopt_matrix(y.reshape(1, -1))
b = cvxopt_matrix(np.zeros(1))

# Setting solver parameters (change default to decrease tolerance) 
cvxopt_solvers.options['show_progress'] = False
cvxopt_solvers.options['abstol'] = 1e-10
cvxopt_solvers.options['reltol'] = 1e-10
cvxopt_solvers.options['feastol'] = 1e-10


# In[172]:


# Run solver
sol = cvxopt_solvers.qp(P, q, G, h, A, b)
alphas = np.array(sol['x'])


# In[174]:


# w parameter in vectorized form
w = ((y * alphas).T @ X).reshape(-1,1)

# Selecting the set of indices S corresponding to non zero parameters
S = (alphas > 1e-4).flatten()

# Computing b
b = y[S] - np.dot(X[S], w)

# Display results
print('Alphas = ',alphas[alphas > 1e-4])
print('w = ', w.flatten())
print('b = ', b[0])


# In[175]:


print(w.shape)


# In[187]:


# test our svm algorithm
error = 0
y_test = y_test.reshape(-1, 1) * 1.
for i,k in enumerate(X_maptest):
    y_pred = np.dot(w.T, np.array(k.T))+b[0]
    if y_pred * y_test[i] < 0:
        error += 1
error


# In[189]:


accuracy_rate = 1 - error / len(y_test)
accuracy_rate


# In[190]:


len(y_test)

# reference for cvxopt: https://xavierbourretsicotte.github.io/SVM_implementation.html
# SVM 中文解释： https://wizardforcel.gitbooks.io/dm-algo-top10/content/svm-3.html

