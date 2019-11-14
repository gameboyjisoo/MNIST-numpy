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
        self._weights = np.random.randn(input2, input1) * np.sqrt(1. / input1)
        self._bias = np.zeros((input2,1))
        self.weight_grad = np.zeros(self._weights.shape)
        self.bias_grad = np.zeros(self._bias.shape)
        self.equation_input = equation_input
        self.activation_result = 0
        if activation_type == "sigmoid":
            self.activation = sigmoid
            self.equation_grad = sigmoid
        elif activation_type == "softmax":
            self.activation = softmax
            self.equation_grad = softmax
        self.activation_grad = 0
        self.equation = np.matmul(self._weights, self.equation_input) + self._bias
    
    def get_weights(self):
        return self._weights
    
    def get_bias(self):
        return self._bias
        
    def set_activation_grad(self, input1, input2):
        if self.activation == sigmoid:
            self.activation_grad = np.matmul(input1, input2)
        
    def set_equation_input(self, inputmatrix):
        self.equation_input = inputmatrix
        
    def set_equation_grad(self):
        if self.activation == sigmoid:
            self.equation_grad = self.activation_grad * sigmoid(self.equation, "backward")
        elif self.activation == softmax:
            self.equation_grad = self.activation_result - Y
    
    def set_weight_grad(self):
        temp_weight_grad = ((1./m_training) * np.matmul(self.equation_grad, self.equation_input.T)) + ((lamda * self._weights) / m_training)
        self.weight_grad = (temp_weight_grad * (1. - momentum) + (self.weight_grad * momentum))
        self._weights = self._weights - learning_rate * self.weight_grad
    
    def set_bias_grad(self):
        temp_bias_grad = (1./m_training) * np.sum(self.equation_grad, axis=1, keepdims=True)
        self.bias_grad = (temp_bias_grad * (1. - momentum) + (self.bias_grad * momentum))
        self._bias = self._bias - learning_rate * self.bias_grad
        
    def forward(self):
        self.equation = np.matmul(self._weights, self.equation_input) + self._bias
        self.activation_result = self.activation(self.equation, "forward")
        return self.activation_result
    
    def backward(self, input1, input2):
        #equation gradient is based on activation grad * sigmoid for sigmoid
        #for softmax it is activation_result - Y
        self.set_activation_grad(input1, input2)
        self.set_equation_grad()
        self.set_weight_grad()
        self.set_bias_grad()
        return self._weights.T, self.equation_grad

# layer inputs 1&2 are used to determine the size of the weights and biases.
# equation_input is what the weights are multiplied by
layer_one = dense_layer(n_x, n_h, X, "sigmoid")
layer_two = dense_layer(n_h, digits, sigmoid(layer_one.equation, "forward"), "softmax")


print("Begin training")

for i in range(300):
    next_input = layer_one.forward()                                                 # layer 1 formula update
    layer_two.set_equation_input(next_input)                                         # layer 2 input update
    next_input = layer_two.forward()                                                 # layer 2 formula update
    loss = cross_entropy_with_regularization(Y, next_input, layer_one.get_weights(), layer_two.get_weights())
    
    next_input, next_input2 = layer_two.backward(0,0)                                # layer 2 update
    layer_one.backward(next_input, next_input2)                                      # layer 1 update
    if (i % 10 == 0):
        print("Epoch", i, "loss: ", loss)

print("Final loss:", loss)
#1000 epoches took slightly less than 7 minutes!


Z1 = np.matmul(layer_one.get_weights(), X_test) + layer_one.get_bias()   #just a simple weight function
A1 = sigmoid(Z1, "forward")                                              #activation function
Z2 = np.matmul(layer_two.get_weights(), A1) + layer_two.get_bias()       #just a simple weight function
A2 = softmax(Z2, "forward")                                              #softmax function


predictions = np.argmax(A2, axis=0)
labels = np.argmax(Y_test, axis=0)

print(confusion_matrix(predictions, labels))
print(classification_report(predictions, labels))
