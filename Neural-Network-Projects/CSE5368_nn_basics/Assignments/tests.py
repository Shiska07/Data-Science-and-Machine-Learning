import numpy as np
# Modify the line below based on your last name
# for example:
from Raut_01_01 import multi_layer_nn
# from Your_last_name_01_01 import multi_layer_nn

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

    return X, y

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

    return X, y

def test_can_fit_data_test():
    np.random.seed(12345)
    from sklearn.model_selection import train_test_split
    X, y = create_toy_data_nonlinear(n_samples=110)
    y = sigmoid(y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    Y_train = Y_train.reshape(-1,1)
    Y_train = Y_train
    Y_test = Y_test.reshape(-1,1)
    Y_test = Y_test

    multi_layer_nn(X_train,Y_train,X_test,Y_test,[2,1],alpha=0.35,epochs=1000,h=1e-8,seed=1234)
    
    return 0

res = test_can_fit_data_test()
