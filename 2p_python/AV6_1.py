# Ejercicio de Samu
import numpy as np

M = 9.3961  # mg

tiempo = np.array([1, 3.25, 5.5, 7.75, 10], dtype=float)
concentracion = np.array([3.6621, 4.5825, 1.8965, 2.6827, 1.9386], dtype=float)

# Por ende, primero deberiamos hallar esta funcion con polinomio interpolador de lagrange

n = len(tiempo)  # no estoy seguro lgm

x = tiempo
y = concentracion

P = np.poly1d([0])

################RUTINA012########################## #LAGRANGE
for i in range(n):
    a = np.delete(np.arange(n), i)
    p = np.poly1d([1, -x[a[0]]])
    for j in range(1, n - 1):
        p = np.polymul(p, np.poly1d([1, -x[a[j]]]))
    P += y[i] * p / p(x[i])

print(f"La funcion de concentracion vs tiempo es: {P}")

A = 0.0  # Area bajo la curva de la funcion concentracion vs tiempo
f = P
a = tiempo[0]
b = tiempo[len(tiempo) - 1]
n = 30  # dato del problema
################RUTINA015########################## ## REGLA DE SIMPSON 1/3
h = (b - a) / n
A = 0
y = f(np.linspace(a, b, n + 1))
k = 0
while k < n:
    A += (h / 3) * (y[k] + 4 * y[k + 1] + y[k + 2])
    k += 2


print(f"b) El Area bajo la curva es es:  {A:0.6f}")


def gastoCardiaco():
    C = (60 * M) / A
    return float(C)


print(f"c) El gasto cardiaco es:  {gastoCardiaco():0.6f}")
