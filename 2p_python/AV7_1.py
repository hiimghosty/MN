# Diferenciacion numerica
import numpy as np


def f(x):
    return np.cos(x)


x = np.pi / 4
h = np.pi / 12
print(f"Derivada real evaluada en el punto de interes: {-np.sin(x):0.6f}")

d1y = (f(x + h) - f(x)) / h
print(f"Primera derivada hacia adelante orden 1: {d1y:0.6f}")
print(f"Eror: {np.abs((-np.sin(x) - d1y) / -np.sin(x))}")
d1y = (-3 * f(x) + 4 * f(x + h) - f(x + 2 * h)) / (2 * h)
print(f"Eror: {np.abs((-np.sin(x) - d1y) / -np.sin(x))}")

print(f"Primera derivada hacia adelante orden 2: {d1y:0.6f}")
print(f"Eror: {np.abs((-np.sin(x) - d1y) / -np.sin(x))}")

d1y = (f(x) - f(x - h)) / h
print(f"Primera derivada hacia atras orden 1: {d1y:0.6f}")
print(f"Eror: {np.abs((-np.sin(x) - d1y) / -np.sin(x))}")

d1y = (3 * f(x) - 4 * f(x - h) + f(x - 2 * h)) / (2 * h)
print(f"Primera derivada hacia atras orden 2: {d1y:0.6f}")
print(f"Eror: {np.abs((-np.sin(x) - d1y) / -np.sin(x))}")

# diferencia central
d1y = (f(x + h) - f(x - h)) / (2 * h)
print(f"Primera derivada central orden 2: {d1y:0.6f}")
print(f"Eror: {np.abs((-np.sin(x) - d1y) / -np.sin(x))}")
d1y = (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h)) / (12 * h)
print(f"Primera derivada central orden 4: {d1y:0.6f}")
print(f"Eror: {np.abs((-np.sin(x) - d1y) / -np.sin(x))}")
