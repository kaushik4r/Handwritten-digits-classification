# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pickle
from scipy.optimize import fmin_cg
import time
import csv
import os

def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return 1.0 / (1.0 + np.exp(-1.0 * z))


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_sample.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     - feature selection"""


    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Pick a reasonable size for validation data

    # ------------Initialize preprocess arrays----------------------#
    train_preprocess = np.zeros(shape=(50000, 784))
    validation_preprocess = np.zeros(shape=(10000, 784))
    test_preprocess = np.zeros(shape=(10000, 784))
    train_label_preprocess = np.zeros(shape=(50000,))
    validation_label_preprocess = np.zeros(shape=(10000,))
    test_label_preprocess = np.zeros(shape=(10000,))
    # ------------Initialize flag variables----------------------#
    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0
    # ------------Start to split the data set into 6 arrays-----------#
    for key in mat:
        # -----------when the set is training set--------------------#
        if "train" in key:
            label = key[-1]  # record the corresponding label
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)  # get the length of current training set
            tag_len = tup_len - 1000  # defines the number of examples which will be added into the training set

            # ---------------------adding data to training set-------------------------#
            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[1000:], :]
            train_len += tag_len

            train_label_preprocess[train_label_len:train_label_len + tag_len] = label
            train_label_len += tag_len

            # ---------------------adding data to validation set-------------------------#
            validation_preprocess[validation_len:validation_len + 1000] = tup[tup_perm[0:1000], :]
            validation_len += 1000

            validation_label_preprocess[validation_label_len:validation_label_len + 1000] = label
            validation_label_len += 1000

            # ---------------------adding data to test set-------------------------#
        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)
            test_label_preprocess[test_len:test_len + tup_len] = label
            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]
            test_len += tup_len
            # ---------------------Shuffle,double and normalize-------------------------#
    train_size = range(train_preprocess.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = train_preprocess[train_perm]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = train_label_preprocess[train_perm]

    validation_size = range(validation_preprocess.shape[0])
    vali_perm = np.random.permutation(validation_size)
    validation_data = validation_preprocess[vali_perm]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = validation_label_preprocess[vali_perm]

    test_size = range(test_preprocess.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = test_preprocess[test_perm]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = test_label_preprocess[test_perm]

    # Feature selection
    # Your code here.
    pixels = 28*28
    removeUninformativeCols = []
    #Select features using the training data and then apply the same selection to validation and test data.
    #I have slightly modified the assignment description (uploaded to ublearns) to reflect this change. 
    #The change is that we need you to submit the list of selected features also as part of the submission.
    #you need to record which features your network uses in the 'params.pickle' file.
    
    backgroundcolor=train_data[0,0]
    for i in range(pixels):
        train = all(x == backgroundcolor for x in train_data[:,i])
        #validate = all(x == backgroundcolor for x in validation_data[:,i])
        #if(train==True and validate==True):
        if(train==True):
            removeUninformativeCols.append(i)
    print(len(removeUninformativeCols))
    train_data = np.delete(train_data, removeUninformativeCols, axis=1)
    validation_data = np.delete(validation_data, removeUninformativeCols, axis=1)
    test_data = np.delete(test_data, removeUninformativeCols, axis=1)
    print('preprocess done')
    

    return train_data, train_label, validation_data, validation_label, test_data, test_label

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))  # each row represnts weight matrix for one hidden node
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0.0

    # Your code here
    Objective = 0.0
    grad_w1 = 0.0
    grad_w2  = 0.0
    trainingDataSize = training_data.shape[0]
    training_data = np.append(training_data,np.ones([len(training_data),1]),1)  #add column
    training_data=training_data.T

    hiddenLayerOutput = np.dot(w1,training_data)  
    hiddenLayerOutput = sigmoid(hiddenLayerOutput)
    
    hiddenOutputIncludingBiasTerm=hiddenLayerOutput.T
    hiddenOutputIncludingBiasTerm = np.append(hiddenOutputIncludingBiasTerm,np.ones([hiddenOutputIncludingBiasTerm.shape[0],1]),1)  #add column  
    output = np.dot(w2,hiddenOutputIncludingBiasTerm.T)
    output=sigmoid(output) #k*1
    outputclass = np.zeros((n_class,training_data.shape[1])) #initialize all output class to 0

    i=0
    for i in range(len(training_label)):
        label=0
        label= int(training_label[i])
        outputclass[label,i] = 1 # set class of true label
    

     #negative log-likelihood error
    Objective += np.sum(outputclass * np.log(output) + (1.0-outputclass) * np.log(1.0-output))
    deltaOutput = output - outputclass  #k*1
    

    #grad_w2 = grad_w2 + (deltaOutput.reshape((n_class,1)) * np.hstack((hiddenLayerOutput,np.ones(1))))
    grad_w2 =  np.dot(deltaOutput.reshape((n_class,training_data.shape[1])), hiddenOutputIncludingBiasTerm)
    outputDeltaSum = np.dot(deltaOutput.T,w2)


    outputDeltaSum = outputDeltaSum[0:outputDeltaSum.shape[0], 0:outputDeltaSum.shape[1]-1]
    delta_hidden = ((1.0-hiddenLayerOutput) * hiddenLayerOutput*outputDeltaSum.T)
    grad_w1 =  np.dot(delta_hidden.reshape((n_hidden,training_data.shape[1])) , (training_data.T))


    Objective = ((-1)*Objective)/trainingDataSize
    randomization = np.sum(np.sum(w1**2)) + np.sum(np.sum(w2**2))
    Objective = Objective + ((lambdaval * randomization) / (2.0*trainingDataSize))
    grad_w1 = (grad_w1 + lambdaval * w1) / trainingDataSize      #equation 16
    grad_w2 = (grad_w2 + lambdaval * w2) / trainingDataSize      #equation 17    
    obj_val = Objective
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    #print(obj_val)
    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = []
    # Your code here
    for testingData in data:
        inputIncludingBiasTerm=np.hstack((testingData,np.ones(1.0)))  #size (d+1)*1
        hiddenLayerOutput = np.dot(w1,inputIncludingBiasTerm)
        hiddenLayerOutput = sigmoid(hiddenLayerOutput)
        
        
        hiddenOutputIncludingBiasTerm=np.hstack((hiddenLayerOutput,np.ones(1.0)))  #size (d+1)*1
        output = np.dot(w2,hiddenOutputIncludingBiasTerm)
        output=sigmoid(output) #k*1
        labels.append(np.argmax(output,axis=0))
        
    labels = np.array(labels)
    # Return a vector with labels   
    return labels




"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50

# set the number of nodes in output unit
n_class = 10
if os.path.exists('ML_Regression.csv'):
    os.remove('ML_Regression.csv')

matrixiteration=[0]

with open('ML_Regression.csv', 'a', newline='\n', encoding='utf-8') as csvFile:
    fieldNames = ['Matrix_iterations','Hidden', 'Lambda', 'Start time', 'End time', 'Execution time', 'train_accuracy', 'validation_accuracy', 'test_accuracy']
    writer = csv.DictWriter(csvFile, fieldnames=fieldNames)
    writer.writeheader()
    for index in matrixiteration:
        if index==0:
            numberofiterations=50
            opts = {'maxiter': 50}  # Preferred value.
        elif index==1:
            numberofiterations=100
            opts = {'maxiter': 100}  # Preferred value.
        elif index==2:
            numberofiterations=150
            opts = {'maxiter': 150}  # Preferred value.'
            
                   
        for number_of_nodes in np.arange(4,104,4):
            n_hidden = number_of_nodes
            n_class = 10
            for lamb in np.arange(0,65,5):
                now_time = time.time()
                start_time = time.strftime("%X")
                lambdaval = lamb
                # initialize the weights into some random matrices
                initial_w1 = initializeWeights(n_input, n_hidden)
                initial_w2 = initializeWeights(n_hidden, n_class)
                
                # unroll 2 weight matrices into single column vector
                initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)
                
                # set the regularization hyper-parameter
                #lambdaval = 0.0
                
                args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)
                
                # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

                nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
  
                # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
                # and nnObjGradient. Check documentation for this function before you proceed.
                #nn_params, cost = fmin_cg(nnObjFunctionVal, x0=initialWeights, fprime=nnObjGradient,args = args, maxiter = 50)
                

                # Reshape nnParams from 1D vector into w1 and w2 matrices
                w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
                w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
                
                
                predicted_label = nnPredict(w1, w2, train_data)           
                # find the accuracy on Training Dataset
                train_accuracy = str(100 * np.mean((predicted_label == train_label).astype(float))) + '%'
                print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')
                
                
                predicted_label = nnPredict(w1, w2, validation_data)
                # find the accuracy on Validation Dataset         
                validation_accuracy = str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%'
                print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')
                
                
                predicted_label = nnPredict(w1, w2, test_data)
                # find the accuracy on Validation Dataset
                test_accuracy = str(100 * np.mean((predicted_label == test_label).astype(float))) + '%'
                print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
                
                
                end_time = time.strftime("%X")
                execution_time = str(round(time.time() - now_time, 2))
                writer.writerow({'Matrix_iterations':numberofiterations,'Hidden': n_hidden, 'Lambda': lambdaval, 'Start time': start_time, 'End time': end_time , 'Execution time': execution_time , 'train_accuracy': train_accuracy, 'validation_accuracy': validation_accuracy, 'test_accuracy': test_accuracy})
                print('Matrix_iterations' + str(numberofiterations) +'number_of_nodes = ' + str(number_of_nodes) + ' and lambda = ' + str(lamb) )
                #obj = [selected_features, n_hidden, w_1, w_2, lambda]
                obj = [n_hidden, w1, w2, lambdaval]
                # selected_features is a list of feature indices that you use after removing unwanted features in feature selection step
                #pickle.dump(obj, open('params.pickle', 'wb'))
