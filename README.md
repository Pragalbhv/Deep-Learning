# Deep Learning Assignment 1: Neural Network development

##### By Pragalbh Vashishtha

This is assignment number 1 in the course, Fundamentals of Deep Learning CS6910 by Prof. Mitesh Khapra.
We implement a feed forward neural network and use verious flavors of the gradient descent algorithm such as momentum, nesterov, RMSprop, Adam, NAdam  and compare  them. The datasets chosen are fashion-mnist and mnist, which have images of size 28x28 and 10 classes. We o not use any convolution,. Instead, we would like to see how a simple 'dense' neural network could perform.

We run upto 450 different configurations and track them all using wandb, we then find correlations with the best features and tune further searches to attempt to reach as high an accuracy as possible:-

Report can be accessed here:- https://wandb.ai/pragalbh/DL-Assign1/reports/-CS6910-Assignment-1--VmlldzozODE3MDAy


### Libraries used:
- copy was used to obtain a deep copy of the class Model
- tqdm was used to track time left in a particular run
- wandb was used to log all the runs with their metrics
- matplotlib and seaborn was used to plot graphs such as confusion matrix, ROC cuvres



### Instructions:

**Dependencies**

python 3.7+
Keras

In addition,  install the following packages (or go `pip install -r requirements.txt`):
- numpy
- tqdm
- wandb
- matplotlib
- copy
- argparse
- keras

## util.py

The utils block contains several functions that are used in the neural network implementation:

  *  The get_activation function returns an activation function based on the input string. It supports the sigmoid, softmax, tanh, and relu activation functions.
       * The sigmoid activation function is a commonly used non-linear function that maps any real-valued number to a value between 0 and 1.
       * The softmax activation function is often used as the final activation function in neural networks for multi-class classification problems. It maps a vector of real numbers to a probability distribution over the classes.
       * The tanh activation function is similar to the sigmoid function but maps the input to a value between -1 and 1.
       * The relu activation function returns the input value if it is positive, and 0 otherwise.

* The get_activation_derivative function returns the derivative of an activation function based on the input string. It supports the sigmoid, softmax, tanh, and relu activation functions. The derivative is used in backpropagation to compute the gradients of the loss with respect to the activations of the previous layer.
	* The sigmoid derivative is computed as the sigmoid activation function multiplied by (1 - sigmoid activation function).
	* The softmax derivative is computed differently because we do not need the full derivative matrix. We only need the diagonal elements because the derivative of the loss with respect to a single output neuron is only affected by that neuron's output. The diagonal elements of the softmax derivative matrix are given by softmax(x_i) * (1 - softmax(x_i)), where x_i is the output of the ith neuron.
	* The tanh derivative is computed as (1 - tanh(x)^2).
	* The relu derivative is computed as 1 if the input is positive and 0 otherwise.

* The get_loss function returns a loss function based on the input string. It supports the cross-entropy and mean squared error loss functions.
	* The cross-entropy loss function is often used for classification problems, especially when the output is a probability distribution over the classes. It measures the difference between the predicted and true probability distributions.

	* The mean squared error (MSE) loss function is commonly used for regression problems. It measures the average squared difference between the predicted and true values.

* The get_loss_derivative function returns the derivative of a loss function based on the input string. It supports the cross-entropy and mean squared error loss functions. The derivative is used in backpropagation to compute the gradients of the loss with respect to the output of the neural network.
	* The MSE loss derivative is computed as 2*(y_pred - y_true), where y_pred and y_true are the predicted and true values, respectively.
	* The cross-entropy loss derivative is computed as -(y_true - y_pred), where y_pred and y_true are the predicted and true probability distributions, respectively.

## preprocess.py
 - one_hot(inarray): This function takes an input array inarray,  and returns a one-hot encoded array outarray. outarray has the same number of rows as inarray and 10 columns, where each row corresponds to one example in the dataset, and the 1 in the corresponding column represents the label of that example.

 - Preprocess(X,y): This function takes two inputs, X and y, which are the feature matrix and label vector for the dataset, respectively. The function first checks if X and y have the same number of examples (rows) and throws an error if they do not match. Then, the function reshapes X into a matrix with dimensions (num_examples, 784), where 784 is the number of pixels in the image. The reshaped matrix is then normalized by dividing each pixel value by 255, which scales the pixel values to a range between 0 and 1. Next, the function transposes the reshaped matrix to a column vector with dimensions (784, num_examples). Finally, the function applies the one_hot() function to the y vector and returns the reshaped X_processed matrix and the one-hot encoded y_processed matrix.

- train_val_split(X, y, splits=0.1): This function takes the X and y matrices and splits them into training and validation sets. The function randomly shuffles the index of the examples and splits them based on the splits parameter, which is the percentage of examples that will be used for the validation set. The function returns the training and validation sets for both X and y.




## arch.py
### class layer
- The code block provides a class called "layer", which is used to create a layer object for neural networks.

- The layer class has several arguments for initialization, including input_size, output_size, activation, batch_size, and type_ (initialization type).

- The input_size argument represents the number of neurons in the previous layer, while the output_size represents the number of neurons in the current layer. The activation argument specifies the activation function for the layer (default is the sigmoid function), and batch_size is the fixed size of batches used for broadcasting (default is 2).

- The type_ argument determines the initialization method for the layer weights. There are four initialization methods available: random, Xavier/Glorot, and He/Kaiming.

- The forward() method computes the forward pass in the layer by multiplying the layer weights with the input and adding biases, followed by applying the activation function.

- The hard_set() method allows the user to input the weight and bias values directly, which is useful for debugging.

### class Model
- This is a Python class named Model which represents a neural network model. 
- The Model class is responsible for defining the structure and behavior of our neural network model. It contains the following attributes:

    - input_size: an integer representing the number of features in the input layer.
    
    - output_size: an integer representing the number of features in the output layer.
    
    - hidden_layer_sizes: a list of integers representing the number of neurons in each of the hidden layers.
    
    - layers: a list of Layer objects representing the layers of our neural network model.
    
    - batch_size: an integer representing the size of the mini-batches we will use for training.

The Model class also has the following methods:

- __init__: a constructor method that initializes the Model object with the given input parameters. It creates the Layer objects for each layer of the network and adds them to the layers list.
- forward: a method that performs the forward pass through the layers of the neural network. It takes an input tensor x as its input, and returns the output tensor of the final layer.
- backward: a method that performs the backward pass through the layers of the neural network. It takes an input tensor x, the target output tensor y, and the predicted output tensor y_pred as its inputs. It computes the gradients of the loss function with respect to the weights and biases of each layer in the network, and stores these gradients in the d_W and d_b attributes of each Layer object.
- predict: a method that makes predictions using the trained neural network. It takes an input tensor Xtest as its input, and returns either the predicted class labels (if probab=False) or the predicted probabilities for each class (if probab=True).

### class optimizer

The optimizers class is initialized with various parameters such as X_size, Y_size, num_layers, const_hidden_layer_size, const_hidden_layer_activation, const_hidden_layer_initializations, loss, optimizer, lamdba, batch_size, epochs, eta, ES, log. The class contains various methods such as __init__, wandb_logger, iterate, loss_calc, loss_calc_fit, accuracy_check.

-  __init__ method initializes the optimizer class with the values of the various parameters as described above. It keeps the batch_size and epochs as class variables, initializes the hidden_layer_sizes, hidden_layer_activations, hidden_layer_initializations, self.model (an object of class Model) using these class variables and other parameters. It initializes train_loss, train_acc, val_loss, val_acc, log, ES, ES_best_val_loss, ES_paitence, ES_model, ES_epoch as instance variables of the class. Here, ES denotes whether early stopping is enabled, and log denotes whether logging is enabled.

-  wandb_logger method logs the training and validation loss and accuracy to wandb, depending on whether the validation data exists or not.


- iterate is a method in the optimizer class that is called by the various optimizer methods in the class, such as batch_gradient_descent, momentum, rmsprop, Adam, Nadam, and NAG. The purpose of iterate is to perform training iterations on the training data, evaluate the loss on both the training and test data after each iteration, and store the losses in the train_loss and test_loss attributes of the optimizer object.

 - The iterate method takes three arguments:

    - updator: a function that performs a weight update on the neural network based on the gradients computed during the forward and backward passes on a mini-batch of training examples
     - X: the input features for the training data
     - Y: the target outputs for the training data

 - iterate method loops over each mini-batch, performs a forward pass, a backward pass, and a weight update using the updator function. After iterating over all the mini-batches (completion of an epoch), it evaluates the loss on both the training and test data using the loss_calc method and stores the losses in the train_loss and test_loss attributes of the optimizer object.
 - loss_calc:

	 - loss_calc is a method in the optimizer class that is called by the iterate method to calculate the loss on the training and test data after each epoch. The purpose of loss_calc is to compute the cross-entropy loss on the specified data set.

	 - The loss_calc method takes two arguments:

		-  X: the input features for the data set
		 - Y: the target outputs for the data set

	 -  loss_calc first performs a forward pass on the data set to obtain the predicted outputs. It then computes the cross-entropy loss between the predicted outputs and the target outputs. Finally, it returns the loss as a scalar value. It also calculates the accuracy

 -  updator:

	 - updator is a function that performs a weight update on the neural network based on the gradients computed during the forward and backward passes on a mini-batch of training examples. The updator function is passed as an argument to the iterate method.

	 - The updator function takes one argument:

		- _: a dummy variable that is not used in the function, used by SGD, batch, Momentum, NAG, RMSprop 
		- t: epoch value, used by optimizers Adam, Nadam

	 -  The updator function updates the weights and biases based on the gradients computed during the backward pass. The specific weight update rule depends on the optimizer being used (e.g. momentum, RMSprop, Adam, NAG).

#### Optimizers

These are the optimizers implemented

- batch_gradient_descent: This function implements Mini-Batch Gradient Descent. It updates the weights and biases of the neural network by taking the average of the gradients of a small batch of data. The size of the batch can be changed by changing the value of batchsize parameter.

- momentum: This function implements Momentum-Based Gradient Descent. It adds a fraction of the previous gradient to the current gradient. This helps the algorithm to avoid oscillations and converge faster.

- rmsprop: This function implements Root Mean Squared Propagation. It scales the learning rate by diving it by the square root of a running average of the squares of the gradients

- Adam: This function implements Adaptive Moment Estimation. It combines the advantages of both momentum-based and rmsprop-based optimization algorithms. It maintains an exponentially decaying average of past gradients and squared gradients, and calculates the update step based on these moving averages.

- NAG: This function implements Nesterov Accelarated Gradient descent. It is a modification of the Momentum-Based Gradient Descent algorithm. It first calculates an approximate position of the next step, then calculates the gradient of the cost function at that position and finally updates the weights and biases of the neural network based on that gradient.

- NAdam: This function implemets the NAdam optimizer. Like Adam, NAdam keeps track of the first and second moments of the gradients, but it also incorporates the Nesterov 'look ahead' idea to update the weights. NAdam also includes bias correction in its update rule, which helps to correct the bias that could be introduced in the initial iterations 

### Class dependencies:-
![classes.png](https://github.com/pragalbhv/Deep-Learning/blob/main/classes.png?raw=true)


## train.py

The train.py script is used to train the neural network using the optimizer class with various options that can be specified using command-line arguments. Here is a description of the available options. The deafult values are set according to what worked best in the wandb sweeps.

    -h, --help: Displays a help message that summarizes the available options and their default values.
    
    -wp WANDB_PROJECT, --wandb_project WANDB_PROJECT: Specifies the name of the Weights & Biases project to log the results to.
    
    -we WANDB_ENTITY, --wandb_entity WANDB_ENTITY: Specifies the entity of the project 
    
    -d DATASET, --dataset DATASET: Specifies the dataset to use for training. The available options are fashion_mnist and mnist, with fashion_mnist being the default.
    
    -e EPOCHS, --epochs EPOCHS: Specifies the number of epochs to train for. The default value is 40.
    
    -b BATCH_SIZE, --batch_size BATCH_SIZE: Specifies the batch size to use for training. The default value is 32.
    
    -l LOSS, --loss LOSS: Specifies the loss function to use for training. The available options are cross_entropy and mean_squared_error, with cross_entropy being the default.
    
    -o OPTIMIZER, --optimizer OPTIMIZER: Specifies the optimization algorithm to use for training. The available options are sgd, momentum, nesterov, rmsprop, adam, and nadam, with nadam being the default.
    
    -lr LEARNING_RATE, --learning_rate LEARNING_RATE: Specifies the learning rate to use for training. The default value is 1e-4.
    -m MOMENTUM, --momentum MOMENTUM: Specifies the momentum to use for nesterov and momentum gradient descent optimization algorithms. The default value is 0.9.
    
    -beta BETA, --beta BETA: Specifies the beta parameter to use for rmsprop optimization algorithm. The default value is 0.9.
    
    -beta1 BETA1, --beta1 BETA1: Specifies the beta1 parameter to use for adam and nadam optimization algorithms. The default value is 0.9.
    
    -beta2 BETA2, --beta2 BETA2: Specifies the beta2 parameter to use for adam and nadam optimization algorithms. The default value is 0.999.
    
    -eps EPSILON, --epsilon EPSILON: Specifies the epsilon parameter to use for rmsprop, adam and nadam optimization algorithm.
    
    -w_d WEIGHT_DECAY, --weight_decay WEIGHT_DECAY: Specifies the L2 regularization parameter to use for training. The default value is 0.
    
    -w_i WEIGHT_INIT, --weight_init WEIGHT_INIT: Specifies the weight initialization method to use for training. The available options are xavier, he, and random, with he being the default.
    
    -nhl NUM_LAYERS, --num_layers NUM_LAYERS: Specifies the number of hidden layers to use for the neural network. The default value is 3.
    
    -sz HIDDEN_SIZE, --hidden_size HIDDEN_SIZE: Specifies the number of neurons to use for each hidden layer. The default value is 512.
    
    -a ACTIVATION, --activation ACTIVATION: Specifies the activation function to use for the neural network. The available options are relu, sigmoid, tanh, and softmax, with relu being the default.
    
    -ES EARLYSTOP, --earlystop EARLYSTOP: Specifies whether or not to perform early stopping during training
