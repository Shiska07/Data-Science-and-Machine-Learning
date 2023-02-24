# Raut, Shiska
# 1001_526_329
# 2023_02_26
# Assignment_01_01
import numpy as np

# returns a weight matrix of size [r, c + 1] (including bias) 
def get_weights_matrix(r, c, seed):
    
    # use random std normal dist to initialize weight values
    np.random.seed(seed)
    weight_matrix = np.random.randn(r, c + 1)   # c + 1 for bias

    return weight_matrix

# given input 'x' and weight matrix 'W' for a single layer, returns logsig(W*x) 
def get_layer_output(x, W):

    # get dot product
    net_value = np.dot(W, x)
    activation_value = 1.0 / (1 + np.exp(-net_value))
    
    return activation_value

# given sample 'x_sample' and list of all weights in the network 'weights_list', 
# returns network output y_sample
def get_network_output(x_sample, weights_list):

    # add 1 to the first row of input 
    nn_input = np.ones((x_sample.shape[0]+1, 1), dtype = float)
    nn_input[1::] = x_sample

    # get output for each layer
    for i in range(len(weights_list)):
        W = weights_list[i]
        layer_output = get_layer_output(nn_input, W)

        # output of the previous layer becomes input of the next layer
        nn_input = np.ones((layer_output.shape[0]+1, 1), dtype = float)
        nn_input[1::] = layer_output
    
    return layer_output

# calculates mean squared error for a single sample
def get_sample_mean_squared_error(y_sample, y_pred):
    
    # get the numer of rows for output value
    ndim_out = y_sample.shape[0]
    
    # sum of squared error
    sse = np.sum(((y_sample - y_pred)**2), axis = 0, dtype = float)
    
    # mean squared error
    mse = sse / ndim_out
    
    return mse[0]

# calculates the average mean squared error for the test data
def get_avg_mse(Y_test, Y_pred):

    # get number of samples and dimension of the output
    ndim_out, n_samples = Y_test.shape

    # calculate mse for each column using vectorization
    mse_each_samp = np.sum((Y_test - Y_pred)**2, axis = 0, dtype = float) / ndim_out

    # calculate average mean sqaured error
    average_mse = np.sum(mse_each_samp, dtype = float) / n_samples

    # return mse as a value not a numpy array
    return average_mse


def adjust_weights(weights_list, x_train_sample, y_train_sample, alpha, h):

    # initialize list to store adjusted weights and temp weights
    adjusted_weights_list = []

    for i in range(len(weights_list)):

        n, m = weights_list[i].shape
        gradient_mtx = np.zeros((n, m), dtype = float)

        # calculate partial derivative wrt each weight in the weight matrix
        for j in range(n):
            for k in range(m):

                # save original value
                orig_val = weights_list[i][j, k].copy()

                # get and store f(x + h) for W(n , m)
                weights_list[i][j, k] = float(orig_val + h)
                y_add_h = get_network_output(x_train_sample, weights_list)
                mse_y_add_h = get_sample_mean_squared_error(y_train_sample, y_add_h)

                # get and store f(x h h) for W(n , m)
                weights_list[i][j, k] = float(orig_val - h)
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
    
    return weights_list


# get predictions for X_test after training
def get_predictions(weights_list, X_test, Y_test):

    n_feat_test, n_test_samples = X_test.shape
    n_out, __ = Y_test.shape

    Y_pred = np.zeros((n_test_samples, n_out), dtype = float)

    for i in range(n_test_samples):
        x_test = X_test[:,i].reshape(n_feat_test, 1)
        y_pred = get_network_output(x_test, weights_list)
        Y_pred[i] = y_pred.squeeze()

    return Y_pred.transpose()


def multi_layer_nn(X_train,Y_train,X_test,Y_test,layers,alpha,epochs,h=0.00001,seed=2):

    # get number of features and number of samples
    n_feat_train, n_train_samples = X_train.shape
    __ , n_test_samples = X_test.shape
    
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

        # record MSE of the test samples per epoch
        avg_mse_per_epoch.append(get_avg_mse(Y_test, Y_pred))

    # get final prediction
    Y_pred_final = get_predictions(weights_list, X_test, Y_test)

    # convert test data average mse list to numpy array
    test_mse = np.array(avg_mse_per_epoch, dtype = float)

    # return final weights, average mse for test per epoch, final prediction
    return weights_list, test_mse, Y_pred_final


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
    assert err[1] < err[0]  # PASSED
    assert err[2] < err[1]  # PASSED
    assert err[3] < err[2]  # PASSED
    assert err[10] < 0.15   # PASSED
    assert err[999] < 0.1   # PASSED
    assert abs(err[9] - 0.10182781417045624) < 1e-5  # FAILED


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
    assert err[1] < err[0]  # PASSED
    assert err[2] < err[1]  # PASSED
    assert err[3] < err[2]  # PASSED
    assert err[10] < 0.04   # PASSED
    assert err[999] < 0.004  # FAILED
    assert abs(err[9] - 0.022177658583431813) < 1e-5 # FAILED


def test_check_weight_update():
    np.random.seed(1234)
    X, y = create_toy_data_nonlinear_2d(110)
    y = sigmoid(y)
    X_train = X[:, :100]
    X_test = X[:, 100:]
    Y_train = y[:, :100]
    Y_test = y[:, 100:]

    np.random.seed(1234)
    [W_before, err, Out] = multi_layer_nn(X_train,Y_train,X_test,Y_test,[2,2],alpha=0.2,epochs=0,h=1e-8,seed=1234)
    np.random.seed(1234)
    [W_after, err, Out] = multi_layer_nn(X_train, Y_train, X_test, Y_test, [2, 2], alpha=0.2, epochs=1, h=1e-8, seed=1234)
    delta1 = (W_after[0] - W_before[0])
    delta2 = (W_after[1] - W_before[1])

    correct_delta1 = np.array([[-6.66044303e-05, -1.51193183e-03, -1.51193183e-03,
                             -1.51193183e-03, -1.51193183e-03],
                            [4.78145648e-04, 1.38747444e-03, 1.38747444e-03,
                             1.38747444e-03, 1.38747451e-03]])
    correct_delta2 = np.array([[-0.00498067, -0.00342466, -0.00417229],
                            [0.00745801, 0.00347394, 0.002611]])

    assert np.allclose(delta1, correct_delta1, atol=1e-5)  # FAILED
    assert np.allclose(delta2, correct_delta2, atol=1e-5)   # FAILED

def test_number_of_nodes_test():
    # check if the number of nodes is being used in creating the weight matrices
    X, y = create_toy_data_nonlinear(n_samples=110)
    X_train = X[:, :100]
    X_test = X[:, 100:]
    Y_train = y[:, :100]
    Y_test = y[:, 100:]

    [W, err, Out] = multi_layer_nn(X_train, Y_train, X_test, Y_test, [
                                   100, 1], alpha=1e-9, epochs=0, h=1e-8, seed=2)

    assert W[0].shape == (100, 5)
    assert W[1].shape == (1, 101)

    [W, err, Out] = multi_layer_nn(X_train, Y_train, X_test, Y_test, [42, 1], alpha=1e-9,
                                   epochs=0, h=1e-8, seed=2)
    assert W[0].shape == (42, 5)
    assert W[1].shape == (1, 43)

    [W, err, Out] = multi_layer_nn(X_train, Y_train, X_test, Y_test, [42, 2], alpha=1e-9,
                                   epochs=0, h=1e-8, seed=2)
    assert W[0].shape == (42, 5)
    assert W[1].shape == (2, 43)

test_number_of_nodes_test()

