# Ejercicio de las bacterias
import numpy as np

np.set_printoptions(suppress=True)
b0 = 0.029  # dato del rpboelma
k = 1.4e-7  # coeficiente dato
B_inicial = 50976.0  # B(0)
x0 = 0  # tiempo inicial
y0 = np.array([B_inicial])


def b(t):
    return b0 * (2 + np.sin(np.pi * (t / 3)))


# Las preguntas me piden hasta 8 segundos, por ende en total necesito 6 puntos
totalPuntos = 8
n_heun = 3
n_hamming = totalPuntos - n_heun
h = 1
F = lambda x, y: b(x) * y - k * (y) ** 2
################RUTINA022- Heun.##########################
#

# rutina:
m = len(y0)
a = np.hstack((np.zeros((m - 1, 1)), np.eye((m - 1))))
resultados = np.zeros((n_hamming + n_heun + 1, 2 + m))
resultados[0, :] = np.concatenate(([0, x0], y0))
x = x0
for i in range(n_heun):
    dy1 = F(x, y0)
    y1 = y0 + h * dy1
    x += h
    dy2 = F(x, y1)
    y = y0 + (h / 2) * (dy1 + dy2)
    y0 = y.copy()
    resultados[i + 1, :] = np.concatenate(([i + 1, x], y0.flatten()))

# ahora hamming
################RUTINA026-Hamming.##########################
#
x0, y0 = resultados[0, 1], resultados[0, 2:].copy()
x1, y1 = resultados[1, 1], resultados[1, 2:].copy()
x2, y2 = resultados[2, 1], resultados[2, 2:].copy()
x3, y3 = resultados[3, 1], resultados[3, 2:].copy()
m = len(y0)
a = np.hstack((np.zeros((m - 1, 1)), np.eye((m - 1))))
x1 = x0 + h
x2 = x1 + h
x3 = x2 + h
for i in range(n_hamming):
    dy1 = F(x1, y1)
    dy2 = F(x2, y2)
    dy3 = F(x3, y3)
    p = y0 + (4 / 3) * h * (2 * dy1 - dy2 + 2 * dy3)
    x4 = x3 + h
    dy4 = F(x4, p)
    y = (9 * y3 - y1) / 8 + (3 / 8) * h * (-dy2 + 2 * dy3 + dy4)
    resultados[n_heun + i + 1, :] = np.concatenate(([n_heun + i + 1, x4], y.flatten()))
    y0 = y1.copy()
    y1 = y2.copy()
    y2 = y3.copy()
    y3 = y.copy()
    x1 = x2
    x2 = x3
    x3 = x4

print(resultados)

# Ítems solicitados
item_a = resultados[2, 2]  # B(2)
item_b = resultados[6, 2]  # B(6)
item_c = b(3)
item_d = b(5)

# Ítems solicitados
item_a = resultados[2, 2]  # B(2)
item_b = resultados[6, 2]  # B(6)
item_c = F(resultados[3, 1], resultados[3, 2])  # B'(3)
item_d = F(resultados[5, 1], resultados[5, 2])  # B'(5)

print(f"Item a - Cantidad de bacterias a los 2 segundos: {item_a:.6f}")
print(f"Item b - Cantidad de bacterias a los 6 segundos: {item_b:.6f}")
print(f"Item c - Índice de crecimiento a los 3 segundos: {item_c:.6f}")
print(f"Item d - Índice de crecimiento a los 5 segundos: {item_d:.6f}")


# a=0 #Puede ser cualquier valor, con tal de crear un espacio de memoria para "a"


################RUTINA013########################## #INTERPOLACION DE NEWTON
n = 9
p = np.poly1d([1, -resultados[0][1]])
P = np.poly1d(resultados[0][2])
for i in range(1, n):
    a = (resultados[i][2] - P(resultados[i][1])) / p(resultados[i][1])
    P += a * p
    p = np.polymul(p, np.poly1d([1, -resultados[i][1]]))

print(f"estimacion para t=4.5 usando el polinomio: {P(4.5):.6f}")
