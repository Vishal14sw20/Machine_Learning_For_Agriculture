import numpy as np


# Usually these parameters are set such that a=1 and b=100
def grad_rosenbrock(z):
    g1 = -400 * z[0] * z[1] + 400 * z[0] ** 3 + 2 * z[0] - 2
    g2 = 200 * z[1] - 200 * z[0] ** 2
    return np.array([g1, g2])


def rosenbrock(z):
    return (1 - z[0]) ** 2 + 100 * (z[1] - z[0] ** 2) ** 2


def grad(z, lr):
    # z = (x,y)
    # initial error
    error = rosenbrock(z)
    iterations = 0
    while error > 0.0000511:
        iterations += 1
        z_new = z - grad_rosenbrock(z) * lr
        error = rosenbrock(z) - rosenbrock(z_new)
        z = z_new

    print("iterations: {} , x,y: {}".format(iterations, z))


grad(np.array([-2, 2]), .0004)


x = np.linespace(-2, 2, 250)
y = np.linespace(-2, 2, 250)
X , Y = np.meshgrid(x, y)
Z = rosenbrock(np.array([X, Y]))
