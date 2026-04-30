import numpy as np

# Constantes
g = 32.17
t = 1.2
x_val = 1.6
tol = 1e-3
n = 100


def f(x):
    return (g / (2 * (x**2))) * (
        ((np.exp(x * t) - np.exp(-x * t)) / (2)) - np.sin(x * t)
    ) - x_val


x0 = -2
x1 = -1

# METODO DE LA SECANTE

################RUTINA008##########################
for i in range(n):
    x2 = (x1 * f(x0) - x0 * f(x1)) / (f(x0) - f(x1))
    err = np.abs(x2 - x1)
    relerr = np.abs(err / x2)
    if tol > err or tol > relerr or tol > np.abs(f(x2)):
        break
    x0 = x1
    x1 = x2


print(f"El valor de w es  {x2:0.6f}")
print("Nro de iteraciones ", i + 1)
print(f"Error absoluto {err:0.6f}")
print(f"Error relativo {relerr:0.6f}")
