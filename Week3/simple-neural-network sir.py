import numpy as np

import matplotlib.pyplot as plt

np.random.seed(1)  # random generator seed


# generate data in interval [-1,+1]

def get_random_x(num_x):
    return np.random.rand(num_x) * 2 - 1


# function we want to approximate

def f_org(x):
    return 1.0 / (np.exp(-10 * x ** 2) + 1) - 0.5

    # return (np.sin(-4*x))


# function f_org with noise:

def f_noisy(x, s=0.1):
    return f_org(x) + s * np.random.randn(len(x))


n = 50  # number of trainings sets

epochs = 1000  # number of epochs

d = 1  # nodes in input layer

k1 = 3  # nodes in hidden layer

k = 1  # nodes in output layer

eta = 0.1  # learning rate

# init parameters

W1 = np.random.rand(k1, d) - 0.5

W2 = np.random.rand(k, k1) - 0.5

b1 = np.random.rand(k1, d) - 0.5

b2 = np.random.rand(k) - 0.5


# activation function (sigmoid):

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# Loss function

def loss(fx, y):
    return (fx - y) ** 2


# forward pass:  f(x) = W2*sigma(W1*x + b1) + b2

def forward(x, W1, W2, b1, b2):
    return np.dot(W2, sigmoid(W1 * x + b1)) + b2


# backward pass: calculate derivatives

def backward(xi, yi, fxi, W1, W2, b1, b2):
    dsig = sigmoid(W1 * xi + b1) * (1 - sigmoid(W1 * xi + b1))

    dw1 = 2 * (fxi - yi) * np.transpose(W2) * dsig * xi

    dw2 = 2 * (fxi - yi) * np.transpose(sigmoid(W1 * xi + b1))

    db1 = 2 * (fxi - yi) * np.transpose(W2) * dsig

    db2 = 2 * (fxi - yi)

    return dw1, dw2, db1, db2


# generate input and output values

x = get_random_x(n)

y = f_noisy(x)

loss_epoch = []

loss_mean = []

li = None

for epoch in range(epochs):

    for i in range(n):
        fxi = forward(x[i], W1, W2, b1, b2)

        li = loss(fxi, y[i])

        loss_epoch.append(li)

        dw1, dw2, db1, db2 = backward(x[i], y[i], fxi, W1, W2, b1, b2)

        # update weights:

        W1 = W1 - eta * dw1

        W2 = W2 - eta * dw2

        b1 = b1 - eta * db1

        b2 = b2 - eta * db2

    lm = np.mean(loss_epoch)

    loss_mean.append(lm)

    loss_epoch = []

plt.figure()

plt.plot(loss_mean, '-o')

plt.title('Mean loss over epochs')

plt.xlabel('epochs')

plt.ylabel('loss')

plt.show()

# ------- generate function for plot ------------------------------------------

nn = 200

fx = []

xx = np.linspace(-1, +1, nn)

for i in range(nn):
    fxi = forward(xx[i], W1, W2, b1, b2)

    fx.append(float(fxi))

plt.figure()

plt.plot(x, y, 'o')

plt.plot(xx, fx, '-')

plt.title('Fitted Function')

plt.show()