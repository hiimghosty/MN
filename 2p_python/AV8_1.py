import numpy as numpy

x0 = 0
y0 = 1
h = 0.1
n = 3


def dy(x, y):
    return 2 * y * x


def d2y(x, y):
    return 2 * y + 2 * x * (2 * x * y)
