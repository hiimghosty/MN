import numpy as np


def f(x):
    return np.exp(x) * ((x) ** 3 + 3)


Area_real = 15 * np.exp(3) - np.exp(1)
xnewton = np.array([1, 1.5, 2, 2.5, 3], dtype=float)

# Primero hacemos una interpolacion de newton
x = xnewton
y = f(xnewton)
n = len(xnewton)
p = np.poly1d([1, -x[0]])
a = 0  # Puede ser cualquier valor, con tal de crear un espacio de memoria para "a"
P = np.poly1d([y[0]])
################RUTINA013########################## #INTERPOLACION DE NEWTON
for i in range(1, n):
    a = (y[i] - P(x[i])) / p(x[i])
    P += a * p
    p = np.polymul(p, np.poly1d([1, -x[i]]))

print(f"Valor para x=2.3 {P(2.3):0.6f}")
# Ahora necesito comparar el area real con el area algebraica del polinomio
A = P.integ()(3) - P.integ()(1)
print(f"Area algebraica usando integ: {A:0.6f}")
errorAbsoluto = abs(A - Area_real)
print(f"Error absoluto: {errorAbsoluto:0.6f}")
# Ahora necesito aproximar la integral con trapecio
a = 1
b = 3
n = len(xnewton) - 1
z = P
################RUTINA014########################## ## REGLA DE TRAPECIO COMPUESTA
h = (b - a) / n
A = 0
y = z(np.linspace(a, b, n + 1))
k = 0
while k < n:
    A += (h / 2) * (y[k] + y[k + 1])
    k += 1
print(f"Area con trapecio: {A:0.6f}")

# error al integrar con trapecio
errorAbsoluto = abs(A - Area_real)
print(f"Error al integrar con trapecio: {errorAbsoluto:0.6f}")
