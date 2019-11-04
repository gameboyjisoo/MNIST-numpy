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

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

class layer():
    def __init__(self, weights, bias, equation_input):
        self.weights = weights
        self.bias = bias
        self.weight_grad = np.zeros(weights.shape)
        self.bias_grad = np.zeros(bias.shape)
        self.equation_input = equation_input
        self.activation = 0
        self.activation_grad = 0
        self.equation_grad = 0
        self.equation = np.matmul(self.weights, self.equation_input) + self.bias
    def forward(self):
        self.equation = np.matmul(self.weights, self.equation_input) + self.bias
        
    def backward(self):
        temp_weight_grad = ((1./m_training) * np.matmul(self.equation_grad, self.equation_input.T)) + ((lamda * self.weights) / m_training)
        self.weight_grad = (temp_weight_grad * (1. - momentum) + (self.weight_grad * momentum))
        self.weights = self.weights - learning_rate * self.weight_grad
        temp_bias_grad = (1./m_training) * np.sum(self.equation_grad, axis=1, keepdims=True)
        self.bias_grad = (temp_bias_grad * (1. - momentum) + (self.bias_grad * momentum))
        self.bias = self.bias - learning_rate * self.bias_grad

    

layer_one = layer(np.random.randn(n_h, n_x) * np.sqrt(1. / n_x),    # weight
                  np.zeros((n_h, 1)),                               # bias
                  X)                                                # equation_input (weight를 뭐랑 곱할지)

layer_two = layer(np.random.randn(digits, n_h) * np.sqrt(1. / n_h), # weight
                  np.zeros((digits, 1)),                            # bias
                  sigmoid(layer_one.equation))                      # equation_input (weight를 뭐랑 곱할지)

for i in range(1000):
    layer_one.forward()
    layer_one.activation = sigmoid(layer_one.equation)   # activation function 1 (sigmoid)
    layer_two.equation_input = layer_one.activation      # layer 2의 input update
    layer_two.forward()                                  # layer 2 수식 update
    layer_two.activation = np.exp(layer_two.equation) / np.sum(np.exp(layer_two.equation), axis=0) # activation function 2 (softmax)
    loss = cross_entropy_with_regularization(Y, layer_two.activation, layer_one.weights, layer_two.weights)
    
    layer_two.equation_grad = layer_two.activation - Y   # cross-entropy & softmax gradient
    layer_two.backward()                                 # layer 2 weights, bias update
    layer_one.activation_grad = np.matmul(layer_two.weights.T, layer_two.equation_grad)                                   # layer 1 activation gradient
    layer_one.equation_grad = layer_one.activation_grad * sigmoid(layer_one.equation) * (1 - sigmoid(layer_one.equation)) # layer 1 equation의 gradient
    layer_one.backward()                                 # layer 2 weights, bias update
    if (i % 10 == 0):
        print("Epoch", i, "loss: ", loss)


print("Final loss:", loss)
#1000번 돌리는데 7분 좀 안되게 걸림!

Z1 = np.matmul(layer_one.weights, X_test) + layer_one.bias #just a simple weight function
A1 = sigmoid(Z1) #activation function
Z2 = np.matmul(layer_two.weights, A1) + layer_two.bias #just a simple weight function
A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0) #softmax function


predictions = np.argmax(A2, axis=0)
labels = np.argmax(Y_test, axis=0)

print(confusion_matrix(predictions, labels))
print(classification_report(predictions, labels))