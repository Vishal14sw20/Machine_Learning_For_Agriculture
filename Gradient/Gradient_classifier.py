import numpy as np


def softmax(X):
    exps = np.exp(X)
    return (exps.transpose() / np.sum(exps, axis=1)).transpose()


def cross_entropy(actual, predicted):
    return -sum([actual[i] * np.log2(predicted[i]) for i in range(len(actual))])


def classify(predicted):
    return np.argmax(predicted, axis=1) + 1


def gd(x, y):
    lr = 0.99
    w_curr = np.array([(0, 0, 0), (0, 0.0, 0.0)])  # check it again
    for i in range(70):
        y_predicted = np.matmul(x, w_curr)
        y_p_probability = softmax(y_predicted)
        Loss = sum(cross_entropy(y, y_p_probability)) / x.shape[0]
        print(Loss)

        #  = x(predicted-actual) its derivative of entropy loss function
        wd = (np.matmul(x.transpose(), y_p_probability - y))
        w_curr = w_curr - wd * lr / x.shape[0]
    return y_p_probability


x = np.array([(-1., 2.5), (-2., 5.), (-1.5, 4.), (-1., 2.3), (-2.5, 6.5), (-1.8, 4.),
              (-1.2, -2.5), (-2.3, -3.), (-1.8, -4.), (-1.9, -2.3), (-2.9, -3.5), (-1.7, -4.),
              (1., -4.5), (0.2, 5.), (0.5, -3.), (1.3, 2.3), (2.5, -1.0), (1.8, 3.)])
y = np.array([(1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0),
              (0, 1, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0),
              (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)])

labels_actual = [np.where(r == 1)[0][0] for r in y]

y_prob = gd(x, y)
labels = classify(y_prob)
