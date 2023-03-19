import numpy as np

####################################################################################################
####################################################################################################
####################################################################################################
'''
Utils Block:-
Includes functions for  calling

-Activation functions
-Activation function derivatives
-Activation function derivatives
-loss functions
-loss function derivatives


'''
####################################################################################################
####################################################################################################
####################################################################################################




def get_activation(activation):
    
    '''
    Computes the activation function
    
    Parameters: activation(str)
    
    Returns: sigmoid(function)/ tanh(function)/ relu(function)
    '''
    def sigmoid(x):
        #STABLE SIGMOID

        return np.where(x >= 0, 
                        1 / (1 + np.exp(-x)), 
                        np.exp(x) / (1 + np.exp(x)))
    def softmax(x):
        #STABLE SOFTMAX
        z=x-np.max(x,axis=0)
        return np.exp(z)/np.sum(np.exp(z),axis=0)
    def relu(x):
        rel=np.where(x >= 0, 
                            x, 
                            0)
        return rel
    if activation=='sigmoid':
        return sigmoid
    elif activation=='softmax':
        return softmax
    elif activation== 'tanh':
        return np.tanh
    elif activation== 'relu':
        return relu

def get_activation_derivative(activation):
    '''
    Computes and returns the activation derivatives. [backprop term d_h/d_a]
    
    Parameters: activation(str)
    
    Returns: sigmoid_d(function)/ tanh_d(function)/ relu_d(function)

    
    '''
    def sigmoid_d(x):
        sig= np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
        return sig*(1-sig)
    def softmax_d(x):
        z=x-np.max(x,axis=0)
        soft=np.exp(z)/np.sum(np.exp(z),axis=0)
        return soft*(1-soft)
    def tanh_d(x):
        return 1-np.tanh(x)**2
    def relu_d(x):
        return np.where(x >= 0, 
                            1, 
                            0)
    
    if activation=='sigmoid':
        return sigmoid_d
    elif activation=='softmax':
        '''
        not required for backprop as we look directly at dL/da_l
        Made for the sake of completeness, and if user wants softmax in the middle. (use cases are not apparent)
        derivative:-
        d S(x_i) /d x_j= S(x_i)*(kronecker delta_i,j -S(x_j))
        But we care about only dh_k,j/da_k,j So no need to implement d S(x_i) /d x_j
        d S(x_i) /d x_i should suffice
        so we get array of [ d S(x_1) /d x_1, d S(x_2) /d x_2, ....]
        
        For MSE loss after softmax, we need cross terms...
        '''
        
        return softmax_d
    elif activation=='tanh':
        return tanh_d
    elif activation=='relu':
        return relu_d
    assert(activation=='relu'or activation=='tanh'or activation=='sigmoid' or activation=='softmax'),\
    'Must be \'relu\'or \'tanh\' or \'sigmoid\' or \'softmax\' '


def get_loss(loss='cross_entropy'):
    '''
    Computes and returns the loss functions. Could be Squared error or cross-entropy
    Parameters: activation(str)
    
    Returns: crossentropy(function)/mean_squared_error(function)
    '''
    
   
    
    safety=1e-30    
    def crossentropy(P,Q):
        assert(P.shape==Q.shape), "Inputs must be of same shape"

        return np.sum([-np.dot(P[:,i],np.log2(Q[:,i]+safety)) for i in range(P.shape[1])])
    def SE(P,Q):
        assert(P.shape==Q.shape), "Inputs must be of same shape"

        return np.sum(np.square(P-Q))
    
    if loss=="mean_squared_error":
        return SE
    return crossentropy
    
    
      
    
    
    
    

def get_loss_derivative(loss):
    '''
    Computes and returns the derivatives of the loss function
    Parameters: activation(str)
    
    Returns: crossentropy_d(function)/SE_d(function)

    '''
    def SE_d(y_in,y_pred_in):
        '''
        derivative of MSE after softmax is used to get probabs from a_L:
        We need indicator because the all terms of y_true are required unlike cross-entropy where only y_pred[l] is required
        Thus transforming the stacked indicator to y_true, not here...
        
        '''

        def indicator(i,j):
                if i==j:
                    return 1
                return 0


        assert(y_in.shape[0]==y_pred_in.shape[0]),"Inputs must contain same number of examples"

        y=y_in.ravel()
        y_pred=y_pred_in.ravel()


        return np.array([
            [2*np.sum([(y_pred[i]-y[i])*y[i]*(indicator(i,j) - y_pred[j]) for i in range(y.shape[0])])]
            for j in range(len(y))
        ])    
   
    
        
    def crossentropy_d(y,y_pred):
        

        return -(y-y_pred)
    
    
    if loss=="cross_entropy":
        return crossentropy_d
    return SE_d
    