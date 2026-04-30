import numpy as np

t = np.array([0, 1, 2], dtype=float)
v = np.array([2, 5, 14], dtype=float)

# Determinar v(t)

m = len(t)
x = t
y = v
################RUTINA011########################## # INTERPOLACION DIRECTA
v = np.vander(x, N=m, increasing=True)
a = np.linalg.solve(v, y)
print(f"Coeficientes del polinomio (menor a mayor) {a}")

P = np.polynomial.Polynomial(a)
print(f"Polinomio es {P}")


# debemos hallar instante en que la velocidad = 10
def f(x):
    return P(x) - 10


x0 = 2
x1 = 1
tol = 1e-4
n = 50
################RUTINA008########################## #METODO SECANTE
for i in range(n):
    x2 = (x1 * f(x0) - x0 * f(x1)) / (f(x0) - f(x1))
    err = np.abs(x2 - x1)
    relerr = np.abs(err / x2)
    if tol > err or tol > relerr or tol > np.abs(f(x2)):
        break
    x0 = x1
    x1 = x2


print(f"El tiempo para v=10 es {x2:0.6f}")


# Ahora pide hallar la distancia recorrida entre t=0 y t=2

# Podriamos integrar por datos tabulados [me suicido]
# O por funciones


################RUTINA014##########################
h = t[1] - t[0]
n = len(t) - 1
A = 0
k = 0
while k < n:
    A += (h / 2) * (y[k] + y[k + 1])
    k += 1

# Entiendo que esto termina al recorrer todo mi vector
# POr ende, directo ya sale ya que me pide de 0 a 2
print(f"Recorre {A:0.6f}")

# Dp pide primera derivada regresiva de 2do orden
f = P
h = t[1] - t[0]
d1y = (3 * f(x) - 4 * f(x - h) + f(x - 2 * h)) / (2 * h)
print(f"La aceleracion en ese punto es: {d1y}")
