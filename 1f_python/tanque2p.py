# EJERCICIO DE LA ESFERA - vaciado de un tanque esferico
# dH/dt = -(C*S*sqrt(2*g*H)) / (pi*(2*R*H - H^2))
import numpy as np

np.set_printoptions(suppress=True)

# --- datos del problema ---
C = 0.75
r = (13.5 / 2) / 100
S = np.pi * r**2  # area del orificio
g = 9.81
Dtanque = 3.0
R = Dtanque / 2  # radio de la esfera = 1.5
H0 = 2.75  # altura inicial


# lado derecho de la EDO (depende SOLO de H -> ecuacion autonoma)
def dHdt(H):
    return -C * S * np.sqrt(2 * g * H) / (np.pi * (2 * R * H - H**2))


y0 = np.array([H0])  # estado inicial [H]
x0 = 0.0  # variable independiente = tiempo t
h = 0.5
n_rk04 = 3
n_milne = 540  # ajustar segun el enunciado (vaciado ~271 s)
################RUTINA023##########################
m = len(y0)
resultados = np.zeros((n_milne + n_rk04 + 1, 2 + m))
resultados[0, :] = np.concatenate(([0, x0], y0))
a = np.hstack((np.zeros((m - 1, 1)), np.eye((m - 1))))
A = lambda x: np.vstack(
    (a, np.array(([0.0]), dtype=float))
)  # <-- A: fila en 0 (no hay parte lineal)
B = lambda x, y: np.vstack(
    (np.zeros((m - 1, 1)), np.array(([dHdt(np.ravel(y)[0])]), dtype=float))
)  # <-- B: toda la EDO, y ahora depende de y

x = x0
for i in range(n_rk04):
    dy0 = A(x) @ y0 + B(x, y0)  # <-- B recibe y
    x += h / 2
    y1 = y0 + (h / 2) * dy0
    dy1 = A(x) @ y1 + B(x, y1)  # <--
    y2 = y0 + (h / 2) * dy1
    dy2 = A(x) @ y2 + B(x, y2)  # <--
    x += h / 2
    y3 = y0 + h * dy2
    dy3 = A(x) @ y3 + B(x, y3)  # <--
    y = y0 + (h / 6) * (dy0 + 2 * dy1 + 2 * dy2 + dy3)
    y0 = y.copy()
    resultados[i + 1, :] = np.concatenate(([i + 1, x], y0.flatten()))
# Ahora, como ya tengo los primeros 3 puntos para mi milne, puedo aplicar milne
a = np.hstack((np.zeros((m - 1, 1)), np.eye((m - 1))))
x0, y0 = resultados[0, 1], resultados[0, 2:].copy()
x1, y1 = resultados[1, 1], resultados[1, 2:].copy()
x2, y2 = resultados[2, 1], resultados[2, 2:].copy()
x3, y3 = resultados[3, 1], resultados[3, 2:].copy()
for i in range(n_milne):
    dy1 = A(x1) @ y1 + B(x1, y1)  # <-- B recibe y
    dy2 = A(x2) @ y2 + B(x2, y2)  # <--
    dy3 = A(x3) @ y3 + B(x3, y3)  # <--
    p = y0 + (4 / 3) * h * (2 * dy1 - dy2 + 2 * dy3)
    x4 = x3 + h
    dy4 = A(x4) @ p + B(x4, p)  # <-- B con el predictor p
    y = y2 + (h / 3) * (dy2 + 4 * dy3 + dy4)
    y0 = y1.copy()
    y1 = y2.copy()
    y2 = y3.copy()
    y3 = y.copy()
    x1 = x2
    x2 = x3
    x3 = x4
    resultados[4 + i, :] = np.concatenate(([4 + i, x3], y3.flatten()))
print(resultados)
# ahi ya es mirar nomas, si pide caudal a los t segundos, hallar la h a ese segundo y reemplazar en q sal
