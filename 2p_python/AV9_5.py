# DECAIMIENTO RADIOACTIVO
import numpy as np

# M(t): cantidad restante de material, en mg
# dM(t)/dt = k * M(t)
# condición inicial: M(0) = 100 mg
# dato adicional: después de 2 años queda el 95 % de la masa inicial
# por lo tanto: M(2) = 95 mg

k = np.log(0.95) / 2  # obtenido analíticamente

h = 0.5
n = 100

x0 = 0.0
y0 = np.array([100.0])
################RUTINA020########################## #EULER
m = len(y0)
resultados = np.zeros((n + 1, 2 + m))
resultados[0, :] = np.concatenate(([0, x0], y0))
a = np.hstack((np.zeros((m - 1, 1)), np.eye((m - 1))))
A = lambda x: np.vstack((a, np.array(([k]), dtype=float)))
B = lambda x: np.vstack((np.zeros((m - 1, 1)), np.array(([0]), dtype=float)))
x = x0
for i in range(n):
    dy = A(x) @ y0 + B(x)
    y = y0 + h * dy
    y0 = y.copy()
    x += h
    resultados[i + 1, :] = np.concatenate(([i + 1, x], y0.flatten()))

print(resultados)
