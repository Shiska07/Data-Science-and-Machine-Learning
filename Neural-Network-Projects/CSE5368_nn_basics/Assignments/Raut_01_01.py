# Raut, Shiska
# 1001_526_329
# 2023_02_26
# Assignment_01_01

import numpy as np

# returns a weight matrix for of size r X c + 1 (including bias) 
def get_weights_matrix(r, c, seed):
    
    np.random.seed(seed)
    weight_matrix = np.random.randn(r, c + 1)   # c + 1 for bias

    return weight_matrix

# given input 'x' and weight matrix 'W' for a single layer, returns logsig(W*x) 
def get_layer_output(x, W):

    net_value = np.dot(W, x)
    activation_value = 1.0 / (1 + np.exp(-net_value))
    
    return activation_value

# given sample 'x_sample' and list of all weights in the network 'weights_list', 
# returns network output y_sample
def get_network_output(x_sample, weights_list):

    nn_input = np.zeros((x_sample.shape[0]+1, 1))
    nn_input[1::] = x_sample

    for i in range(weights_list):
        W = weights_list[i]
        layer_output = get_layer_output(nn_input, W)
        next_layer_input = np.zeros((layer_output.shape[0]+1, 1))
        next_layer_input[1::] = layer_output
    
    return layer_output

# cakculates mean squared error for a single sample
def get_sample_mean_squared_error(y_sample, y_pred_sample):
    
    # get the numer of rows for output value
    n = y_sample.shape[0]
    
    # sum of squared error
    sse = np.sum(((y_sample - y_pred_sample)**2), axis = 1)
    
    # mean squared error
    mse = sse/n
    
    return mse

# calculates the average mse for results of a single epoch
def get_average_mse(Y, Y_pred):
    pass

# adjusts weights after a training sample has been processed
def adjust_weights(weights_list, x_train_sample, alpha, h):
    pass

# a single pass over the training data
def train_network(X_train,Y_train, weights_list, alpha, h):

    # get number of features and number of samples
    n_feat_train, n_train = X_train.shape

    for i in range(n_train):

        # take a single sample from training data 
        x_sample =  X_train[:,i]
        y_sample = get_network_output(x_sample, weights_list)

    pass

# get predictions for X_test after training
def get_predictions(weights_list, X_test):
    pass


def multi_layer_nn(X_train,Y_train,X_test,Y_test,layers,alpha,epochs,h=0.00001,seed=2):
    
    # reshape input matrices/vectors
    X_train = np.transpose(X_train)
    Y_train = np.transpose(Y_train)
    X_test = np.transpose(X_test)
    Y_test = np.transpose(Y_test)

    # get number of features and number of samples
    n_feat_train, n_train = X_train.shape
    n_feat_test , n_test = X_test.shape
    
    #get number of layers and initialize list to store weights for each layer
    n_layers = len(layers)
    weights_list = []
    
    # initalize list to store average MSE per epoch
    avg_mse_per_epoch = []
    
    # get weights for each layer
    for i in range(n_layers):
        
        if i == 0:
            # get weights for first layer
            weights_list.append(get_weights_matrix(layers[0], n_feat_train, seed))
        else:
            n_nodes = layers[i]
            n_input = layers[i - 1]
            weights_list.append(get_weights_matrix(n_nodes, n_input, seed))
    
    # do the following per epoch
    
    weights_list = train_network(X_train, Y_train, weights_list, alpha, h)
    # Y_pred = get_predictions(weights_list, X_test)
    # avg_mse_per_epoch.append(get_average_mse(Y_test, Y_pred))
        
    # alpha: learning rate
    # epochs: number of epochs for training.
    # h: step size
    # seed: random number generator seed for initializing the weights.
    # return: This function should return a list containing 3 elements:
        # The first element of the return list should be a list of weight matrices.
        # Each element of the list corresponds to the weight matrix of the corresponding layer.

        # The second element should be a one dimensional array of numbers
        # representing the average mse error after each epoch. Each error should
        # be calculated by using the X_test array while the network is frozen.
        # This means that the weights should not be adjusted while calculating the error.

        # The third element should be a two-dimensional array [output_dimensions,nof_test_samples]
        # representing the actual output of network when X_test is used as input.

    # Notes:
    # DO NOT use any other package other than numpy
    # Bias should be included in the weight matrix in the first column.
    # Assume that the activation functions for all the layers are sigmoid.
    # Use MSE to calculate error.
    # Use gradient descent for adjusting the weights.
    # use centered difference approximation to calculate partial derivatives.
    # (f(x + h)-f(x - h))/2*h
    # Reseed the random number generator when initializing weights for each layer.
    # i.e., Initialize the weights for each layer by:
    # np.random.seed(seed)
    # np.random.randn()
    pass