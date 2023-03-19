# Deep Learning Assignment 1: Neural Network development

This is assignment number 1 in the course, Fundamentals of Deep Learning CS6910 by Prof. Mitesh Khapra.
We implement a feed forward neural network and use verious flavors of the gradient descent algorithm such as momentum, nesterov, RMSprop, Adam, NAdam  and compare  them. The datasets chosen are fashion-mnist and mnist, which have images of size 28x28 and 10 classes. We o not use any convolution,. Instead, we would like to see how a simple 'dense' neural network could perform.

We run upto 450 different configurations and track them all using wandb, we then find correlations with the best featuires and tune further searches to attempt to reach as high an accuracy as possible:-

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

## arch.py
### class layer
- The code block provides a class called "layer", which is used to create a layer object for neural networks.

- The layer class has several arguments for initialization, including input_size, output_size, activation, batch_size, and type_ (initialization type).

- The input_size argument represents the number of neurons in the previous layer, while the output_size represents the number of neurons in the current layer. The activation argument specifies the activation function for the layer (default is the sigmoid function), and batch_size is the fixed size of batches used for broadcasting (default is 2).

- The type_ argument determines the initialization method for the layer weights. There are four initialization methods available: random, Xavier/Glorot, and He/Kaiming.

- The forward() method computes the forward pass in the layer by multiplying the layer weights with the input and adding biases, followed by applying the activation function.

- The hard_set() method allows the user to input the weight and bias values directly, which is useful for debugging.

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
