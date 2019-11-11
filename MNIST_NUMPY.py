# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 18:12:38 2019

@author: Jisoo
"""

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784')
X, y = mnist["data"], mnist["target"]

X = X / 255

digits = 10
examples = y.shape[0] #this is where the "answer" is written. If it is a 5, then the corresponding index is 5

y = y.reshape(1, examples)

Y_new = np.eye(digits)[y.astype('int32')] #eye-diagonals are 1, else is 0
# make it 1 only for the corresponding slot of the number (if 4, then index 5 is 1, else is 0)
Y_new = Y_new.T.reshape(digits, examples)

m_training = 60000
m_test = X.shape[0] - m_training

X_train, X_test = X[:m_training].T, X[m_training:].T
Y_train, Y_test = Y_new[:,:m_training], Y_new[:,m_training:]

shuffle_index = np.random.permutation(m_training)
X_train, Y_train = X_train[:, shuffle_index], Y_train[:, shuffle_index]


n_x = X_train.shape[0]
n_h = 64
momentum = 0.9
learning_rate = 1
lamda = 0.01

X = X_train
Y = Y_train


#cross entropy loss
def cross_entropy_with_regularization(Y, Y_hat, weight_one, weight_two):

    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    cross_entropy = -(1/m) * L_sum
    
    regularization = 0.5 * lamda * (1./m_training) * (np.sum(np.square(weight_one)) + np.sum(np.square(weight_two)))

    return cross_entropy + regularization

#sigmoid and sigmoid gradient
def sigmoid(z, direction):
    if direction == "forward":
        s = 1 / (1 + np.exp(-z))
        return s
    else:
        s = sigmoid(z, "forward") * (1-sigmoid(z, "forward"))
        return s

#softmax and softmax gradient
def softmax(z, direction):
    if direction == "forward":
        s = np.exp(z) / np.sum(np.exp(z), axis=0)
        return s
    else:
        return 0


class dense_layer():
    def __init__(self, input1, input2, equation_input, activation_type):
        self.weights = np.random.randn(input2, input1) * np.sqrt(1. / input1)
        self.bias = np.zeros((input2,1))
        self.weight_grad = np.zeros(self.weights.shape)
        self.bias_grad = np.zeros(self.bias.shape)
        self.equation_input = equation_input
        self.activation_result = 0
        if activation_type == "sigmoid":
            self.activation = sigmoid
            self.equation_grad = sigmoid
        elif activation_type == "softmax":
            self.activation = softmax
            self.equation_grad = softmax
        self.activation_grad = 0
        self.equation = np.matmul(self.weights, self.equation_input) + self.bias
        
    def set_equation_input(self, inputmatrix):
        self.equation_input = inputmatrix
        
    def set_equation_grad(self, calculated):
        if self.activation == sigmoid:
            self.equation_grad = calculated * sigmoid(self.equation, "backward")
        elif self.activation == softmax:
            self.equation_grad = self.activation_result - calculated
        
    def forward(self):
        self.equation = np.matmul(self.weights, self.equation_input) + self.bias
        self.activation_result = self.activation(self.equation, "forward")
        return self.activation_result
        
    def backward(self, calculated):
        
        #equation gradient is based on activation grad * sigmoid for sigmoid
        #for softmax it is activation_result - Y
        self.set_equation_grad(calculated)
        
        temp_weight_grad = ((1./m_training) * np.matmul(self.equation_grad, self.equation_input.T)) + ((lamda * self.weights) / m_training)
        self.weight_grad = (temp_weight_grad * (1. - momentum) + (self.weight_grad * momentum))
        self.weights = self.weights - learning_rate * self.weight_grad
        
        temp_bias_grad = (1./m_training) * np.sum(self.equation_grad, axis=1, keepdims=True)
        self.bias_grad = (temp_bias_grad * (1. - momentum) + (self.bias_grad * momentum))
        self.bias = self.bias - learning_rate * self.bias_grad



# layer inputs 1&2 are used to determine the size of the weights and biases.
# equation_input is what the weights are multiplied by
layer_one = dense_layer(n_x, n_h, X, "sigmoid")
layer_two = dense_layer(n_h, digits, sigmoid(layer_one.equation, "forward"), "softmax")


for i in range(300):
    next_input = layer_one.forward()                                                 # layer 1 formula update
    layer_two.set_equation_input(next_input)                                         # layer 2 input update
    next_input = layer_two.forward()                                                 # layer 2 formula update
    loss = cross_entropy_with_regularization(Y, next_input, layer_one.weights, layer_two.weights)
    
    layer_two.backward(Y)                                                                                         # layer 2 weights, bias update
    layer_one.activation_grad = np.matmul(layer_two.weights.T, layer_two.equation_grad)                           # layer 1 activation gradient
    layer_one.backward(layer_one.activation_grad)                                                                 # layer 1 weights, bias update
    if (i % 10 == 0):
        print("Epoch", i, "loss: ", loss)


print("Final loss:", loss)
#1000 epoches took slightly less than 7 minutes!

Z1 = np.matmul(layer_one.weights, X_test) + layer_one.bias #just a simple weight function
A1 = sigmoid(Z1, "forward") #activation function
Z2 = np.matmul(layer_two.weights, A1) + layer_two.bias #just a simple weight function
A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0) #softmax function


predictions = np.argmax(A2, axis=0)
labels = np.argmax(Y_test, axis=0)

print(confusion_matrix(predictions, labels))
print(classification_report(predictions, labels))
