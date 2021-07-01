import numpy as np
import matplotlib.pyplot as plt


def analytic_weights(X, y):
    # Also called closed form solution
    # Taking the derivative of the loss with respect to  w  and setting it equal to 0 gives the analytic solution:

    # actually x_transpose is x and x is x_transpose

    X_transpose = X.transpose()
    multiplication = X.dot(X_transpose)
    inverse = np.linalg.inv(multiplication)
    final = inverse.dot(X)
    w_star = final.dot(y)
    return w_star
    ## end of function


def gradient_descent(x, y, iterations):
    b_curr = 0
    w_curr = np.array([0, 0])
    n = len(x.transpose())
    learning_rate = 0.059
    cost_list = list()

    for i in range(iterations):
        # y_predicted = np.matmul(w_curr, x) + b_curr
        y_predicted = np.matmul(w_curr, x)
        cost = (1 / n) * sum([val ** 2 for val in (y - y_predicted)])
        wd = -(2 / n) * np.matmul(x, (y - y_predicted))

        # bd = -(2 / n) * sum(y - y_predicted)
        w_curr = w_curr - learning_rate * wd
        # b_curr = b_curr - learning_rate * bd
        # print("m {}, b {}, cost {} iteration {}".format(w_curr, b_curr, cost, i))
        cost_list.append(cost)
        print("m {}, cost {} iteration {}".format(w_curr, cost, i))
    fig, axes = plt.subplots()
    axes.scatter(range(iterations), cost_list)
    plt.ylabel("Loss Values")
    plt.show()

# x = np.array([[1, 2, 3, 4, 5], [5, 5, 1, 2, 3]])  # ideal values
x = np.random.randn(100).reshape(2, 50)
y = np.random.randn(50)
# y = np.array([5, 7, 9, 11, 13]) ideal

gradient_descent(x, y, 50)
print("closed form solution: {}".format(analytic_weights(x, y)))
print("BOOM !!!")
