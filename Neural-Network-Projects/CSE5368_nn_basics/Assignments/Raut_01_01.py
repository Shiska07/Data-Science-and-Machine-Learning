# Raut, Shiska
# 1001_526_329
# 2023_02_26
# Assignment_01_01

import numpy as np

# returns a weight matrix of size [r, c + 1] (including bias a the first column) 
def get_weights_matrix(r, c, seed):
    
    # use random std normal dist to initialize weight values
    np.random.seed(seed)
    weight_matrix = np.random.randn(r, c + 1)   # c + 1 for bias

    return weight_matrix


# given input column vector 'x' and weight matrix 'W' of single layer, returns logsig(W*x) 
def get_layer_output(x, W):

    # get dot product
    net_value = np.dot(W, x)
    activation_value = 1.0 / (1 + np.exp(-net_value))
    
    return activation_value


# given sample 'x_sample' and list of all weights in the network 'weights_list', 
# returns network prdicted value 'y_pred'
def get_network_output(x_sample, weights_list):

    # add 1 to the first row of input 
    nn_input = np.ones((x_sample.shape[0]+1, 1), dtype = float)
    nn_input[1::] = x_sample

    # calculate output of the entire network by computing one layer at a time
    for i in range(len(weights_list)):

        W = weights_list[i]

        # let output of a single layer
        layer_output = get_layer_output(nn_input, W)

        # output of the previous layer becomes input of the next layer
        nn_input = np.ones((layer_output.shape[0]+1, 1), dtype = float)
        nn_input[1::] = layer_output
    
    # output from the final layer is the predicted value
    y_pred = layer_output

    return y_pred


# calculates mean squared error for a single sample 
# 'y_sample' & 'y_pred' are 1D column vectors
def get_sample_mean_squared_error(y_sample, y_pred):
    
    # get the numer of output values per sample
    ndim_out = y_sample.shape[0]
    
    # calculate mean squared error for sample
    # x = 0 sums over all rows
    mse = np.sum(((y_sample - y_pred)**2), axis = 0, dtype = float) / ndim_out

    # mse is a (1, 1) array so just return the numerical value
    return mse[0]


# calculates the average mean squared error for the entire test data
# 'Y_test' & 'Y_pred' are 2D matrices, each column represents a sample datapoint
def get_avg_mse(Y_test, Y_pred):

    # get number of samples and dimension of the output
    ndim_out, n_samples = Y_test.shape

    # calculate mean squared error for each sample
    # x = 0 sums over all rows
    mse_each_samp = np.sum((Y_test - Y_pred)**2, axis = 0, dtype = float) / ndim_out

    # calculate average mean sqaured error 
    # by adding all values and dividing by number of samples
    average_mse = np.sum(mse_each_samp, dtype = float) / n_samples

    return average_mse


# adjusts weights of the network given a single training sample
def adjust_weights(weights_list, x_train_sample, y_train_sample, alpha, h):

    # initialize list to store adjusted weights
    adjusted_weights_list = []

    # for each weight matrix
    for i in range(len(weights_list)):

        n, m = weights_list[i].shape

        # initalize matrix to store partial derivative values w.r.t. each weight
        gradient_mtx = np.zeros((n, m), dtype = float)

        # for each weight element 
        for j in range(n):
            for k in range(m):

                # save original value
                orig_val = weights_list[i][j, k].copy()

                # calculate network mse after updating weight (W[j, k] = w + h)
                # keep all other weights constant
                weights_list[i][j, k] = float(orig_val + h)
                y_pred_add_h = get_network_output(x_train_sample, weights_list)
                mse_add_h = get_sample_mean_squared_error(y_train_sample, y_pred_add_h)

                # calculate network mse after updating weight (W[j, k] = w - h)
                # keep all other weights constant
                weights_list[i][j, k] = float(orig_val - h)
                y_pred_sub_h = get_network_output(x_train_sample, weights_list)
                mse_sub_h = get_sample_mean_squared_error(y_train_sample, y_pred_sub_h)
                
                # restore original value in the weight matrix
                weights_list[i][j, k] = orig_val
                
                # calculate and save partial derivative value
                gradient_mtx[j, k] = (mse_add_h - mse_sub_h) / (2*h)

        # new_weight = old_weight - alpha*gradient
        new_wt_mtx = weights_list[i] - (alpha*gradient_mtx)

        # add weight matrix to th list of adjusted weights
        adjusted_weights_list.append(new_wt_mtx)

    return adjusted_weights_list
    

# a single pass over the training data
def train_network(X_train,Y_train, weights_list, alpha, h):

    # get number of features, number of samples and dimension of the output
    n_feat_train, n_train_samples = X_train.shape
    n_out_train, __ = Y_train.shape

    # for each training sample
    for i in range(n_train_samples):
 
        x_sample =  X_train[:,i].reshape(n_feat_train, 1)
        y_sample = Y_train[:,i].reshape(n_out_train, 1)

        # adjust weights using centered difference approximation 
        # to calculate partial derivatives
        weights_list = adjust_weights(weights_list, x_sample, y_sample, alpha, h)
    
    # return updates weights list
    return weights_list


# get predictions for test data after training
def get_predictions(weights_list, X_test, Y_test):

    # get number of features, number of samples and dimension of the output for test data
    n_feat_test, n_test_samples = X_test.shape
    n_out, __ = Y_test.shape

    # initialize an array to store predicted values fot the entire test dataset
    Y_pred = np.zeros((n_test_samples, n_out), dtype = float)

    # get prediction value for each test sample
    for i in range(n_test_samples):
        x_test = X_test[:,i].reshape(n_feat_test, 1)
        y_pred = get_network_output(x_test, weights_list)
        Y_pred[i] = y_pred.squeeze()

    return Y_pred.transpose()


def multi_layer_nn(X_train,Y_train,X_test,Y_test,layers,alpha,epochs,h=0.00001,seed=2):

    # get number of features and number of samples
    n_feat_train, n_train_samples = X_train.shape
    
    #get number of layers and initialize list to store weights for each layer
    n_layers = len(layers)
    weights_list = []
    
    # initalize list to store average MSE per epoch
    avg_mse_per_epoch = []
    
    # initialize weights for each layer
    for i in range(n_layers):
        
        if i == 0:
            # get weights for first layer
            weights_list.append(get_weights_matrix(layers[0], n_feat_train, seed))
        else:
            n_nodes = layers[i]
            n_input = layers[i - 1]
            weights_list.append(get_weights_matrix(n_nodes, n_input, seed))
    
    # do the following per epoch
    for i in range(epochs):

        # train network
        weights_list = train_network(X_train, Y_train, weights_list, alpha, h)

        # test network
        Y_pred = get_predictions(weights_list, X_test, Y_test)

        # record average mea squared error for the test data
        avg_mse_per_epoch.append(get_avg_mse(Y_test, Y_pred))

    # get final prediction
    Y_pred_final = get_predictions(weights_list, X_test, Y_test)

    # convert test data average mse list to numpy array
    avg_test_mse_per_epoch = np.array(avg_mse_per_epoch, dtype = float)

    # return final weights, average mse for test per epoch, final prediction
    return weights_list, avg_test_mse_per_epoch, Y_pred_final

