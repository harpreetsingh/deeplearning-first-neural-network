import numpy as np

# activation function
def sigmoid (x):
    return 1 / ( 1 + np.exp(-x))

# derivative of activation function
def sigmoid_prime (x):
    return sigmoid (x) * (1 - sigmoid(x))

learnrate = 0.5
x = np.array ([1, 2])
y = np.array (0.5)

# initial weights
w = np.array ([0.5, -0.5])

print ("** Simple Gradient Descent: 1 Step Neural Network ** ")
print ("** Build a forward pass function **")
print ("initial wt:", w)
print ("Input:", x)
print ("Expected output:", y)

# calculate gradient descent
# gradient descent is the dot product of inputs*weight muted
# by the sigmoid function
y_hat = sigmoid (np.dot(x, w))
print ("Calculated output: ", y_hat)

error = y - y_hat
print ("Error:", error)

# error_term = (y - y_hat) * derivative_activation_func (h)
# error_term = (y - y_hat) * derivative_activation_func (sigma(wi*xi))
error_term = error * sigmoid_prime (np.dot(x, w))
print ("Error Term:", error_term)

# gradient descent
# delta wi = learningrate * error_term * xi
delta_w = learnrate * error_term * x
print ("gradient descent:", delta_w)
