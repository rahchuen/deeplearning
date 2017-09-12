"""
N-layer neural network 
Input -> (N-1) layers [z_l/Relu] -> 1 layer [z_N/Sigmoid] -> Output
where z_l = W_l*a_(l-1) + b_l for layer l
Cost function = cross entropy
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
import numpy as np
import scipy
from utils import *

# calculates sigmoid of vector Z
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

# calculates relu of vector Z
def relu(Z):
    A = np.maximum(0,Z)
    cache = Z
    return A, cache

# sigmoid back prop dL/dZ = dL/dA * dA/dZ
def sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)   
    return dZ

# relu back prop
def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True) 
    dZ[Z <= 0] = 0   
    return dZ

# computes cross-entropy cost between predictions AL and labels Y
def compute_cost(AL, Y):
    m = Y.shape[1] # number of training examples
    cost = (-1.0/m)*(np.dot(Y,np.log(AL).T) + np.dot((1-Y),(np.log(1-AL).T)))
    return cost

# initialize Weights Wi and Biases bi for all layers based on list containing dimensions of each layer l
def initialize_parameters(layer_dims):
    np.random.seed()
    parameters = {}
    num_layers = len(layer_dims) # number of layers in the network

    for l in range(1, num_layers): 
        # Wi -- weight matrix (layer_dims[i], layer_dims[i-1])
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) 
        #bi -- bias vector (layer_dims[i], 1)
        parameters['b'+ str(l)] = np.zeros((layer_dims[l],1))

    return parameters  

# Implement the linear part of forward propagation
# Z = W.X + b
def linear_forward(A, W, b):
    Z = np.dot(W,A) + b
    cache = (A, W, b)
    return Z, cache

# forward propagation for one layer including activation function
# returns A and cache of linear prop (A,X,b) and activation cache (Z)
def forward_propagation_single_layer(A_prev, W, b, activation): 
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev,W,b) # Z = WX+b
        A, activation_cache = sigmoid(Z) # A=sigmoid(Z)
  
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = relu(Z)
    
    cache = linear_cache, activation_cache

    return A, cache 

# Does forward propagation for whole model - linear->relu for L-1 layers and linear->sigmoid last
def forward_propagation_model(X, parameters):
    caches = [] # initialize cache list
    A = X 
    num_layers = len(parameters) // 2   

    # Linear -> Relu for num_layers-1 times 
    for l in range(1, num_layers):
        A_prev = A
        W = parameters['W' + str(l)] # get initialized parameters for layer l
        b = parameters['b' + str(l)]
        A, cache = forward_propagation_single_layer(A_prev, W, b, activation = "relu")
        caches.append(cache) # cache[l] contains linear cache and activation cache for layer l
    
    # Final layer, linear -> sigmoid 
    AL, cache = forward_propagation_single_layer(A, parameters['W'+str(num_layers)], parameters['b'+str(num_layers)],                                    activation = "sigmoid")
    caches.append(cache)
            
    return AL, caches

# linear part of backward propagation for layer l
# all derivatives are wrt loss function (dwi = dL/dwi)
# cache is from forward propagation in the current layer l
def linear_backward(dZ, cache):
    # Z = w*A_prev + b
    # dL/dZ = dZ
    # dL/dW = dL/dZ * dZ/dA = dZ * A_prev
    # dL/db = dL/dZ * dZ/dB = dZ 
    # dL/dA_prev = dL/dZ * dZ/dA_prev = dZ*W
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1.0/m)* np.dot(dZ,A_prev.T)
    db = (1.0/m) * np.sum(dZ,axis=1,keepdims=True)
    dA_prev = np.dot(W.T,dZ)

    return dA_prev, dW, db

def backward_propagation_single_layer(dA, cache, activation):
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ,linear_cache)
    
    return dA_prev, dW, db

# Backward propagation for the linear-relu layers L-1 times then for the last linear->sigmoid layer
# AL is the prediction vector, output of sigmoid(Z) at the last layer
# Y are labels
# caches contain all linear (A,W,b) and activation (Z) caches from fwd prop at each layer l,
# from cache[0] to cache[L-1] 
# cache[l] from 0 to L-2 contains caches of forward_propagation for L-1 relu layers
# cache[L-1] contains cache of forward_propagation for sigmoid layer
def backward_propagation_model(AL, Y, caches):
    grads = {}
    num_layers = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) 
    
    # Initializing the backpropagation dL/dAL (L is cross entropy loss function)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # back prop for Lth layer (SIGMOID -> LINEAR) gradients
    current_cache = caches[num_layers-1]
    grads["dA" + str(num_layers)], grads["dW" + str(num_layers)], grads["db" + str(num_layers)] = \
            backward_propagation_single_layer(dAL, current_cache, activation = "sigmoid")
    
    # back prop for (L-1)th to 1st layer: (RELU -> LINEAR) gradients.
    for l in reversed(range(num_layers-1)): #L-2,L-1,..1
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = backward_propagation_single_layer(grads["dA"+str(l+2)], current_cache, activation = "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

# updates the parameters dWi and dbi
def update_params(parameters, grads, learning_rate):
    num_layers = len(parameters) // 2 # number of layers in the neural network
    for l in range(num_layers):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - (learning_rate*grads["dW"+str(l+1)])
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - (learning_rate*grads["db"+str(l+1)])
        
    return parameters

# This function is used to predict the results of a  L-layer neural network.
def predict(X, y, parameters):    
    m = X.shape[1] # m training examples
    num_layers = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m)) # initialize probability vector
    
    probs, caches = forward_propagation_model(X, parameters)

    for i in range(0, probs.shape[1]):
        if probs[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    print np.sum(p==y)
    print "accuracy = " + str(np.sum(p==y).astype(float)/m)
    return p

# final function to run n layer model
# returns parameters Wi and bi
def run_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    np.random.seed(1)
    costs = []                         
    
    # Parameters initialization.
    parameters = initialize_parameters(layers_dims)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = forward_propagation_model(X, parameters)
        
        # Compute cost.
        cost = compute_cost(AL, Y)

        # Backward propagation.
        grads = backward_propagation_model(AL, Y, caches)

        # Update parameters.
        parameters = update_params(parameters, grads, learning_rate)
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
        
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters
