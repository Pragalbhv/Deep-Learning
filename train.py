import numpy as np
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import copy

from utils import *
from arch import *
from preprocess import*

import argparse

from keras.datasets import fashion_mnist
from keras.datasets import mnist

def main(args):
    
        run=wandb.init(config=args)
        X_size=784 #same for both datasets
        Y_size=10 #same for both datasets
        
        num_layers=wandb.config.num_layers
        const_hidden_layer_size=wandb.config.hidden_size
        const_hidden_layer_activation=wandb.config.activation.lower() #convert string to lower case for ease
        const_hidden_layer_initializations=wandb.config.weight_init.lower()#convert string to lower case for ease
        loss=wandb.config.loss
        optimizer=wandb.config.optimizer.lower()
        lamdba=wandb.config.weight_decay
        batch_size=wandb.config.batch_size
        epochs=wandb.config.epochs
        eta=wandb.config.learning_rate
        ES=wandb.config.earlystop
        log=wandb.config.logger
        dataset=wandb.config.dataset.lower()
        
        
        momentum=wandb.config.momentum
        beta=wandb.config.beta
        beta1=wandb.config.beta1
        beta2=wandb.config.beta2
        epsilon=wandb.config.epsilon
        
        
        
        
        
        
        
        ####################################################### 
        if dataset=='fashion_mnist':
            (X_train,y_train),(X_test,y_test)=fashion_mnist.load_data()
        else: #dataset=='mnist'
            (X_train,y_train),(X_test,y_test)=mnist.load_data()
            
            
        Xtest,ytest=Preprocess(X_test,y_test)
        
        opt=optimizers(X_size,Y_size,num_layers,const_hidden_layer_size,const_hidden_layer_activation,\
                       const_hidden_layer_initializations,loss,optimizer,lamdba,batch_size,epochs,eta,ES,log)
        
        if wandb.config.mode.lower()=='test':
            Xtrainfull,ytrainfull=Preprocess(X_train,y_train) ####Used for TESTING. DO NOT pass this to the model  FOR TRAINING!!!!
            opt.run((Xtrainfull,ytrainfull),momentum=momentum,beta=beta,beta1=beta1,beta2=beta2,epsilon=epsilon)
        else:
            
            Xtrain,Xval,ytrain,yval=train_val_split(X_train,y_train,splits=0.1) # use this when training
            Xtrain,ytrain=Preprocess(Xtrain,ytrain)
            Xval,yval=Preprocess(Xval,yval)
            opt.run((Xtrain,ytrain),(Xval,yval),momentum=momentum,beta=beta,beta1=beta1,beta2=beta2,epsilon=epsilon)
            
            
        
            
            
        return opt.model.predict(Xtest, wandb.config.probab)

        
        
        
        
        
        
        

         
        
         
         
    ###################################################################Argepass Block#######################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-wp',"--wandb_project",type=str,default="DL-Assign1")
    parser.add_argument("-we","--wandb_entity",type=str,default="pragalbh")
    parser.add_argument("-d","--dataset",type=str,default='fashion_mnist',help="Datasets: fashion_mnist or mnist")
    parser.add_argument("-e","--epochs",type=int,default=40,help='Number of epochs')
    parser.add_argument("-b","--batch_size",type=int,default=32,help='Batch Size')
    parser.add_argument("-l","--loss",type=str,default="cross_entropy",help="Cross-entropy loss/ Mean Squared Error loss")
    parser.add_argument("-o","--optimizer",type=str,default="nadam",help="sgd/momentum/nesterov/rmsprop/adam/nadam")
    parser.add_argument("-lr","--learning_rate",type=float,default=1e-4,help="Learning Rate")
    parser.add_argument("-m","--momentum",type=float,default=0.9,help="Momentum used by nesterov & momentum gd")
    parser.add_argument("-beta","--beta",type=float,default=0.9,help="Beta used by rmsprop")
    parser.add_argument("-beta1","--beta1",type=float,default=0.9,help="Beta1 used by adam & nadam")
    parser.add_argument("-beta2","--beta2",type=float,default=0.999,help="Beta2 used by adam & nadam")
    parser.add_argument("-eps","--epsilon",type=float,default=1e-10)
    parser.add_argument("-w_d","--weight_decay",type=float,default=0,help="L2 Regularizer")
    parser.add_argument("-w_i","--weight_init",type=str,default="He",help="Xavier, He or Random Initialization")   
    parser.add_argument("-nhl","--num_layers",type=int,default=3,help="Number of hidden layers in the network")
    parser.add_argument("-sz","--hidden_size",type=int,default=512,help="Number of neurons in each hidden layer")
    parser.add_argument("-a","--activation",type=str,default='relu',help="Activation Function")
    parser.add_argument("-ES","--earlystop",type=bool,default=True,help="Perform Early Stopping or not")
    parser.add_argument("-lg","--logger",type=bool,default=True,help="Log to wandb or not")
    parser.add_argument("-md","--mode",type=str,default="test",help="Test mode, or train+val")   
    parser.add_argument("-prb","--probab",type=bool,default=True,help="Test mode, or train+val")   
    


args = parser.parse_args()

main(args)  