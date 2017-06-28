import numpy as np
from numpy import array, dot, random
import math

# unit step function is used here, can be changed
def activation_function(weighted_sum_inputs):
    if weighted_sum_inputs >=0:
        return 1
    else:
        return 0

# predict takes two vectors, inputs and weights, and calculates the dot product first. Then if the activation function \
# is applied to result to predict.
def predict(inputs,weights):
    sum_weights = np.dot(inputs,weights[1:])+weights[0]
    prediction = activation_function(sum_weights)
    return prediction

def calculate_error(actual,predict):
    return actual-predict

def initialize_weights(trainingdata):
    return np.random.rand(len(training_data[0][0])+1)

def adjust_weights(weights,error,inputs,learningrate):
    # bias
    weights[0] += learningrate*error
    for i in range(1,len(weights)):
        weights[i] += learningrate*error*inputs[i-1]
    return weights

def fit(trainingdata,learningrate,minmse):
    weights = initialize_weights(trainingdata)
    print "initial weights= " 
    print weights
    epoch = 0
    mse = 1000.0
    while (mse > minmse):
        error = 0
        print "epoch = %d" %epoch
        
        for i in range(0,len(trainingdata)):
            inputs = training_data[i][0]
            target = training_data[i][1]
            print "target = %s" %target
            print "inputs = %s" %inputs
            
            prediction = predict(inputs,weights)
            print "prediction = %s" %prediction
            err = calculate_error(target,prediction)
            print err
            error += abs(err)

            weights = adjust_weights(weights,err,inputs,learningrate)
            
        mse = round(round(error,6)/float(len(trainingdata)),6)
        print "error = %s for %s training data" %(error,len(trainingdata))
        print "mse of epoch %s is %s" %(epoch,mse)
        
        epoch+=1
    return weights
training_data = [ (array([0,0,1]), 0), (array([0,1,1]), 1), (array([1,0,1]), 1), (array([1,1,1]), 1),                (array([1,0,1]), 1)]

print fit(training_data,.1,0.0001)

