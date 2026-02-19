import numpy as np

# Resolver x^(3/2) + x = 3
# Definimos la f(x) = 0 como f(x) = x^(3/2) + x - 3 = 0


def f(x):
    return (
        x ** (3 / 2) + x - 3
    )  # Como el enunciado no nos da a ni b, por teorema de bolzano f(a)*f(b) < 0, tanteamos


a = 1.0
b = 2.0
c0 = a

tol = 1e-6
n = 3


###############RUTINA004########################## ##BISECCION
for i in range(n):
    c = (a + b) / 2
    err = np.abs(c0 - c)
    relerr = np.abs(err / c)
    if tol > err or tol > relerr or tol > np.abs(f(c)):
        break
    else:
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    c0 = c


print(c0)
