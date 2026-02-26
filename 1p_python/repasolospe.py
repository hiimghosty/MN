# Ejercicio del plano inclinado y eso
import numpy as np

x_val = 1.7
t = 1
g = 32.17
# Tenemos de dato una x(t), pero en realidad queremos hallar la w para esto


def f(x):
    return (g / (2 * (np.pow(x, 2)))) * (
        -(np.exp(x * t) - np.exp(-x * t)) / 2 + np.sin(x * t)
    ) - x_val


a = -0.5
b = -1.5
print(f(a) * f(b))

tol = 1e-5
n = 100
c0 = a
################RUTINA005########################## #FALSA POSICION
for i in range(n):
    c = (b * f(a) - a * f(b)) / (f(a) - f(b))
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


print(f" El valor de w es: {c:0.6f}")
print(f"El error absoluto es: {err:0.6f}")
print(f"El error relativo es: {relerr:0.6f}")
print(f"F(c) es: {f(c):0.6f}")
print("\n Nro de iteraciones: ", i + 1)
