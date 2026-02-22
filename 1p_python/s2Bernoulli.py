import numpy as np

Q = 1.2  # volume rate
g = 9.81
b0 = 1.8  # width of channel
h0 = 0.6  # upstream water level
H = 0.075  # height of the bump
n = 100

a = -1
b = 2
c0 = a


# Si la funcion llega a una c =0, hace division por 0, eso es kk
def f(x):
    return (
        ((Q**2) / (2 * g * (b0**2) * (h0**2)))
        + h0
        - ((Q**2) / (2 * g * (b0**2) * (x**2)))
        - x
        - H
    )


tol = 1e-6
################RUTINA004##########################
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
print("\n La raiz es ", c)
