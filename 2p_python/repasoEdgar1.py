import numpy as np


def f(x):
    return -0.1 * (np.pow((x - 5), 2)) + 1 + np.sqrt(x - 1)


def g(x):
    return 1.2 - np.sqrt(0.6 + (x - 10) ** 2)


def k(x):
    return 1.1 - np.sqrt(0.6 + (x - 2.5) ** 2)


def p(x):
    return -1.2 + np.sqrt(6 + (x - 6.2) ** 2)


Apiedra = 0.38
velocida = 2.21
