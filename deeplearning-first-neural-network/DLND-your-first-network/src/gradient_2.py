import numpy as np
from data_prep import features, targets, features_test, targets_test

#print ("features test: \n", features_test.head(5))

# activation function
def sigmoid (x):
    return 1 / ( 1 + np.exp(-x))

# derivative of activation function
def sigmoid_prime (x):
    output = sigmoid (x)
    return output * (1 - ouput)

np.random.seed(42)
n_records, n_features = features.shape

# print (n_records, n_features)

# Initialize weights
weights = np.random.normal(scale=1 / n_features**.5, size=n_features)

# randomized weights
#print (weights)

# Neural Network hyperparameters
epochs = 1000
learnrate = 0.5

for e in range(epochs):
    del_w = np.zeros(weights.shape)
    for x, y in zip(features.values, targets):
        # Loop through all records, x is the input, y is the target

        # Activation of the output unit
        output = sigmoid(np.dot(x, weights))

        # The error, the target minues the network output
        error = y - output

        # The gradient descent step, the error times the gradient times the inputs
        del_w += error * output * (1 - output) * x

        # Update the weights here. The learning rate times the
        # change in weights, divided by the number of records to average
    weights += learnrate * del_w / n_records

    # Printing out the mean square error on the training set
    last_loss = 0
    if e % (epochs / 10) == 0:
        out = sigmoid(np.dot(features, weights))
        loss = np.mean((out - targets) ** 2)
        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
tes_out = sigmoid(np.dot(features_test, weights))
print (tes_out)
predictions = tes_out > 0.5
print (predictions)
print (targets_test)
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))
