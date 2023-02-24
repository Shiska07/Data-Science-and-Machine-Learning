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

    # add 1 to the first row of input 
    nn_input = np.ones((x_sample.shape[0]+1, 1))
    nn_input[1::] = x_sample.reshape(x_sample.shape[0], 1)

    # get output for each layer
    for i in range(len(weights_list)):
        W = weights_list[i]
        layer_output = get_layer_output(nn_input, W)

        # output of the previous layer becomes input of the next layer
        nn_input = np.ones((layer_output.shape[0]+1, 1))
        nn_input[1::] = layer_output
    
    return layer_output

# cakculates mean squared error for a single sample
def get_sample_mean_squared_error(y_sample, y_pred):
    
    # get the numer of rows for output value
    n_out = y_sample.shape[0]
    
    # sum of squared error
    sse = np.sum(((y_sample - y_pred)**2), axis = 1)
    
    # mean squared error
    mse = sse/n_out
    
    return mse


def adjust_weights(weights_list, x_train_sample, y_train_sample, alpha, h):

    # initialize list to store adjusted weights and temp weights
    adjusted_weights_list = []

    for i in range(len(weights_list)):

        n, m = weights_list[i].shape
        gradient_mtx = np.zeros((n, m))

        # calculate partial derivative wrt each weight in the weight matrix
        for j in range(n):
            for k in range(m):

                # save original value
                orig_val = weights_list[i][j, k]

                # get and store f(x + h) for W(n , m)
                weights_list[i][j, k] = orig_val + h
                y_add_h = get_network_output(x_train_sample, weights_list)
                mse_y_add_h = get_sample_mean_squared_error(y_train_sample, y_add_h)

                # get and store f(x h h) for W(n , m)
                weights_list[i][j, k] = orig_val - h
                y_sub_h = get_network_output(x_train_sample, weights_list)
                mse_y_sub_h = get_sample_mean_squared_error(y_train_sample, y_sub_h)
                
                # restore original value in the weight matrix
                weights_list[i][j, k] = orig_val
                
                # calculate and save gradiet value
                gradient_mtx[j, k] = (mse_y_add_h - mse_y_sub_h) / (2*h)

        # new_weight = old_weight - alpha*gradient
        new_wt_mtx = weights_list[i] - (alpha*gradient_mtx)

        adjusted_weights_list.append(new_wt_mtx)

    return adjusted_weights_list
    

# a single pass over the training data
def train_network(X_train,Y_train, weights_list, alpha, h):

    # get number of features and number of samples
    n_feat_train, n_train_samples = X_train.shape
    n_out_train, __ = Y_train.shape

    for i in range(n_train_samples):

        # take a single sample from training data 
        x_sample =  X_train[:,i].reshape(n_feat_train, 1)
        y_sample = Y_train[:,i].reshape(n_out_train, 1)
        weights_list = adjust_weights(weights_list, x_sample, y_sample, alpha, h)
    pass


# calculates the average mse for results of a single epoch
def get_average_mse(Y_test, Y_pred):
    pass

# get predictions for X_test after training
def get_predictions(weights_list, X_test):
    pass


def multi_layer_nn(X_train,Y_train,X_test,Y_test,layers,alpha,epochs,h=0.00001,seed=2):

    # get number of features and number of samples
    n_feat_train, n_train_samples = X_train.shape
    __ , n_test_samples = X_test.shape
    
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
    return 0

def sigmoid(x):
    # This function calculates the sigmoid function
    # x: input
    # return: sigmoid(x)
    # Your code goes here
    return 1/(1+np.exp(-x))

def create_toy_data_nonlinear(n_samples=1000):
    X = np.zeros((n_samples, 4))
    X[:, 0] = np.linspace(-1, 1, n_samples)
    X[:, 1] = np.linspace(-1, 1, n_samples)
    X[:, 2] = np.linspace(-1, 1, n_samples)
    X[:, 3] = np.linspace(-1, 1, n_samples)

    y = X[:, 0]**2 + 2*X[:, 1]  - 0.5*X[:, 2] + X[:, 3]**3 + 0.3

    # shuffle X and y
    idx = np.arange(n_samples)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    return X.T, y[:, np.newaxis].T

def create_toy_data_nonlinear_2d(n_samples=1000):
    X = np.zeros((n_samples, 4))
    X[:, 0] = np.linspace(-1, 1, n_samples)
    X[:, 1] = np.linspace(-1, 1, n_samples)
    X[:, 2] = np.linspace(-1, 1, n_samples)
    X[:, 3] = np.linspace(-1, 1, n_samples)

    y = np.zeros((n_samples, 2))
    y[:, 0] = 0.5*X[:, 0] -0.2 * X[:, 1]**2 - 0.2*X[:, 2] + X[:, 3]*X[:,1] - 0.1
    y[:, 1] = 1.5 * X[:, 0] + 1.25 * X[:, 1]*X[:, 0] + 0.4 * X[:, 2] * X[:, 0]

    # shuffle X and y
    idx = np.arange(n_samples)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    return X.T, y.T

def test_can_fit_data_test():
    np.random.seed(12345)
    X, y = create_toy_data_nonlinear(n_samples=110)
    y = sigmoid(y)
    X_train = X[:, :100]
    X_test = X[:, 100:]
    Y_train = y[:, :100]
    Y_test = y[:, 100:]

    [W, err, Out] = multi_layer_nn(X_train,Y_train,X_test,Y_test,[2,1],alpha=0.35,epochs=1000,h=1e-8,seed=1234)
    assert err[1] < err[0]
    assert err[2] < err[1]
    assert err[3] < err[2]
    assert err[10] < 0.15
    assert err[999] < 0.1
    assert abs(err[9] - 0.10182781417045624) < 1e-5



def test_can_fit_data_test_2d():
    np.random.seed(1234)
    X, y = create_toy_data_nonlinear_2d(110)
    y = sigmoid(y)
    X_train = X[:, :100]
    X_test = X[:, 100:]
    Y_train = y[:, :100]
    Y_test = y[:, 100:]

    [W, err, Out] = multi_layer_nn(X_train,Y_train,X_test,Y_test,[2,2],alpha=0.35,epochs=1000,h=1e-8,seed=1234)
    #print(err[10], err[999], err[9])
    assert err[1] < err[0]
    assert err[2] < err[1]
    assert err[3] < err[2]
    assert err[10] < 0.04
    assert err[999] < 0.004
    assert abs(err[9] - 0.022177658583431813) < 1e-5

test_can_fit_data_test()

