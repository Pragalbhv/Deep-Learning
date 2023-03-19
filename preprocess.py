import numpy as np
####################################################################################################
####################################################################################################
####################################################################################################
'''
Preproccess Block:-

Includes functions for 

-One hot
-Preprocessing which includes reshaping, normalizing transposing, one-hotting,
-Activation function derivatives
-loss functions
-loss function derivatives


'''
####################################################################################################
####################################################################################################
####################################################################################################

def one_hot(inarray): 
    '''
    Converts input to one hot array
    '''
    outarray = np.zeros((inarray.size, 10))
    outarray[np.arange(inarray.size), inarray] = 1
    return outarray

def Preprocess(X,y):
      
    '''Unrolls X,y, rehsapes into column vectors, one hots y'''
    assert(X.shape[0]==y.shape[0]),"Inputs must contain same number of examples, stored in rows" #checks if same dim
    
    X_processed=np.reshape(X,(X.shape[0],784))/255 #reshaping and normalizing
    X_processed=X_processed.T #transposing
    y_processed=one_hot(y).T #one hotting
    return np.array(X_processed),y_processed
        

def train_val_split(X, y, splits=0.1):
    '''
    Splits data in train and validation sets
    '''
    i = int((1 - splits) * X.shape[0])         
    index = np.random.permutation(X.shape[0])

    Xtrain, Xval = np.split(np.take(X,index,axis=0), [i])
    ytrain, yval = np.split(np.take(y,index), [i])
    return Xtrain, Xval, ytrain, yval
