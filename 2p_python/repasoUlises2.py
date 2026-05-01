import numpy as np


def F(x):
    return 2.3 * x + np.sqrt(x)


AreaDelAgujero = 231  # m
f = F
a = 0
b = 9.4
n = 15
################RUTINA014##########################
A = 0.0
h = (b - a) / n
for xi in np.arange(a, b, h):
    A = A + (h / 2) * (f(xi) + f(xi + h))
print(f"El area que cubre la chapa es: {A:0.6f}")
print(
    f"a) La cantidad de chapas necesarias para cubrir el agujero es: {AreaDelAgujero / A:0.6f}"
)
chapa_sobrante = ((2 * A - AreaDelAgujero) / (2 * A)) * 100
print(f"b) Chapa sobrante: {chapa_sobrante:0.6f} %")
# El item C no quiero hacer
