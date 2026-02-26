import numpy as np

# Constantes
x_val = 1.7
t = 1
g = 32.17


# Definicion de funcion f(x) = 0
def x_pos(w):
    return (g / (2 * (w**2))) * (np.sin(w * t) - (np.exp(w * t) - np.exp(-w * t)) / 2)


def f(x):
    return x_pos(x) - x_val


# Parametros de inicializacion
n = 100
tol = 1e-5
a = -0.5
b = -1.5
c0 = a
print("el valor de f(a) * f(b) = ", f(a) * f(b))
# %%  RUTINA005##########################
for i in range(n):
    c = (b * f(a) - a * f(b)) / (f(a) - f(b))
    err = np.abs(c0 - c)
    relerr = err / np.abs(c)
    if tol > err or tol > relerr or tol > np.abs(f(c)):
        break
    else:
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    c0 = c

print("Analisis de Convergencia: ")
print("error absoluto: ", err)
print("error relativo: ", relerr)
print("el valor de f(c): ", np.abs(f(c)))
print("numero de iteraciones: ", i + 1)

print("----------------")
print("a) Cantidad de iteraciones: ", i + 1)
print("b) El valor de w es: ", np.round(c, 6))
print("c) El error absoluto: ", np.round(err, 6))
print("d) El error relativo: ", np.round(relerr, 6))
