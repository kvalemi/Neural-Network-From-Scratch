## Libraries ##
import numpy as np 
import pandas as pd 
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import itertools
from sklearn.utils import shuffle


## NN class
class dlnet:
    
    ## Defining the class attributes
    def __init__(self, x, y):
        # input layer
        self.X = x
        
        # desired output
        self.Y = y
        
        # network output
        self.Yh = np.zeros((1, self.Y.shape[1]))
        
        # number of layers
        self.L = 2
        
        # the number of nodes (neurons) in each layer
        # indices correspond to the nodes in each respective layer
        self.dims = [9, 15, 1]
        
        # dictionariy to hold W and b parameters of the network
        self.param = {}
        
        # cache for intermediary values
        self.ch = {}
        
        self.grad = {}
        # loss value of the network
        self.loss = []
        
        # learning rate        
        self.lr = 0.07
        
        # number of training samples
        self.sam = self.Y.shape[1]
        
        # confidence threshold
        self.threshold = 0.5
        
        
    ## NN class functions ##
        
        
    ## NN Parameter initialization function 
    def nInit(self):    
        np.random.seed(1)

        # number of rows is the number of hidden units of that layer 
        # and number of rows from previous layer
        self.param['W1'] = np.random.randn(self.dims[1], self.dims[0]) / np.sqrt(self.dims[0]) 

        # same number of rows as W1 and a single column
        self.param['b1'] = np.zeros((self.dims[1], 1)) 

        # number rows is the number of hidden nodes in the current layer 
        # and the columns is the number of columns of the previous layer
        self.param['W2'] = np.random.randn(self.dims[2], self.dims[1]) / np.sqrt(self.dims[1]) 

        # same number of rows as W2 and a single column
        self.param['b2'] = np.zeros((self.dims[2], 1))  

        return
    

    ## Forward pass functions
    
    # Forward Function
    def forward(self):    
        Z1 = self.param['W1'].dot(self.X) + self.param['b1'] 
        A1 = Relu(Z1)
        self.ch['Z1'],self.ch['A1']=Z1,A1

        Z2 = self.param['W2'].dot(A1) + self.param['b2']
        A2 = Sigmoid(Z2)
        self.ch['Z2'],self.ch['A2']=Z2,A2

        self.Yh = A2
        loss = self.nloss(A2)
        
        return self.Yh, loss

    
    ## Loss calculation function
    def nloss(self,Yh):
        loss = (1./self.sam) * (-np.dot(self.Y,np.log(Yh).T) - np.dot(1-self.Y, np.log(1-Yh).T))    
        return loss
    
    
    ## Back propogation functions
    
    def backward(self):
        dLoss_Yh = - (np.divide(self.Y, self.Yh ) - np.divide(1 - self.Y, 1 - self.Yh))        

        dLoss_Z2 = dLoss_Yh * dSigmoid(self.ch['Z2'])    
        dLoss_A1 = np.dot(self.param["W2"].T,dLoss_Z2)
        dLoss_W2 = 1./self.ch['A1'].shape[1] * np.dot(dLoss_Z2,self.ch['A1'].T)
        dLoss_b2 = 1./self.ch['A1'].shape[1] * np.dot(dLoss_Z2, np.ones([dLoss_Z2.shape[1],1])) 

        dLoss_Z1 = dLoss_A1 * dRelu(self.ch['Z1'])        
        dLoss_A0 = np.dot(self.param["W1"].T,dLoss_Z1)
        dLoss_W1 = 1./self.X.shape[1] * np.dot(dLoss_Z1,self.X.T)
        dLoss_b1 = 1./self.X.shape[1] * np.dot(dLoss_Z1, np.ones([dLoss_Z1.shape[1],1]))  

        self.param["W1"] = self.param["W1"] - self.lr * dLoss_W1
        self.param["b1"] = self.param["b1"] - self.lr * dLoss_b1
        self.param["W2"] = self.param["W2"] - self.lr * dLoss_W2
        self.param["b2"] = self.param["b2"] - self.lr * dLoss_b2
            
            
    ## Iteration Functions     
    def gd(self,X, Y, iter = 3000):
        np.random.seed(1)                         

        self.nInit()

        for i in range(0, iter):
            Yh, loss=self.forward()
            self.backward()

            if i % 500 == 0:
                print ("Cost after iteration %i: %f" %(i, loss))
                self.loss.append(loss)

        return
    
    
    ## Prediction Function
    def pred(self,x, y):  
        self.X=x
        self.Y=y
        comp = np.zeros((1,x.shape[1]))
        pred, loss= self.forward()    
    
        for i in range(0, pred.shape[1]):
            if pred[0,i] > self.threshold: comp[0,i] = 1
            else: comp[0,i] = 0
    
        mis_rate = 1 - np.sum((comp == y)/x.shape[1])

        print("Misclassification Rate: ", mis_rate)
        
        return mis_rate


## Utility Functions ##

# Sigmoid Function
def Sigmoid(Z):
    return 1/(1 + np.exp(-Z))

# Relu Function
def Relu(Z):
    return np.maximum(0,Z)

def dRelu(x):
    x[x<=0] = 0
    x[x>0]  = 1

    return x

def dSigmoid(Z):
    s = 1/(1 + np.exp(-Z))
    dZ = s * (1-s)
    return dZ





    