# Your name : Valinda Vanam
# Your UCM ID : 700703487
# Certificate of Authenticity: “I certify that the codes/answers of this assignment are entirely my own work.”

import sys
from itertools import repeat

import numpy as np

# array to store the training dataset
x_train = []
y_train = []

# loading training files and save the content in two arrays
dataset = open(sys.argv[-2], 'r')
data = dataset.read().splitlines()
for d in data:
    int_data = list(map(float, d.split()))
    y_train.append(int_data.pop())
    x_train.append(int_data)

# array to store the testing dataset
x_test = []
y_test = []

# loading testing files and save the content in two arrays
dataset = open(sys.argv[-1], 'r')
data = dataset.read().splitlines()
for d in data:
    int_data = list(map(float, d.split()))
    y_test.append(int_data.pop())
    x_test.append(int_data)

# displaying the number of rows and columns
print(f"The length of training dataset and testing dataset is {len(x_train)} and {len(x_test)}")


# function to calculate sigmoid value
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# function to calculate the error value
def cost(y, h):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


# gradient descent values calculating function
def gradient(x, y, h):
    return np.dot(x.T, (h - y)) / len(y)


# initializing the parameters
epochs = 10000
alpha = 0.1
weights = list()
x = np.insert(x_train, 0, 1, axis=1)
classes = np.unique(y_train)

# training modules for all the classes
for outputClass in classes:
    y = (y_train == outputClass).astype(int)  # considering one class at a time
    weight = np.zeros(x.shape[1])  # initializing the weights
    error = np.zeros(epochs)
    tempError = 10000
    threshold = 2
    i = 0
    while i < epochs:  # iterating over epochs
        h = sigmoid(np.dot(x, weight))
        error[i] = cost(y, h)  # calculating cost function
        
#cost of current function is less than previous one, then loop should repeat, else stop
#cost means error

        weight = weight - (alpha * gradient(x, y, h))  # updating weights
        i += 1
        if error[i] < tempError:
                tempError = error[i]
                if tempError <= threshold:
                        break

    weights.append(weight)

# displaying final weights
print("The outputs of the training weights are")
for j, w in enumerate(weights):
    print("Class ID%02d: %02d" % (j + 1, j))  # displaying class id
    for i, th in enumerate(w):
        print(f"\u03F4{i + 1} = %0.4f" % th)  # displaying class weights


# function to predict the class
def predict(xtest, weights, classes):
    x = np.insert(xtest, 0, 1, axis=1)
    ypred = []
    for x_text in x:
        temp = [sigmoid(np.dot(x_text, w)) for w in weights]
        ypred.append(classes[np.argmax(temp)])
    return ypred


# predicting and printing the results
predictedOutput = predict(x_test, weights, classes)
predictedError = [1 if i == j else 0 for i, j in zip(y_test, predictedOutput)]
for i, data in enumerate(zip(y_test, predictedOutput, predictedError)):
    print("ID = %05d" % i, "Predicted class = %02d" % data[0], "True class = %02d" % data[1], "Accuracy = %04d" % data[2])
