# EJERCICIO DE YUTU DE ULISES
# Ejercicio del desplazamiento de un ciclista en
# una montanha
# dy/dx = x^2 * y
import numpy as np

y0 = np.array([1.0])
x0 = 0.0
h = 0.1
n_rk04 = 3
n_milne = 7  # para completar 1000m

################RUTINA023##########################
m = len(y0)
# filas: suficientes para todos los pasos del tiempo (3 de rk04 y los restantes milne)
# columnas: una para i, otra para la x, y las restantes para los valores de y
resultados = np.zeros((n_milne + n_rk04 + 1, 2 + m))
# iteracion, x0 , y0. Separo de y0 xq y0 ya es un array
resultados[0, :] = np.concatenate(([0, x0], y0))
a = np.hstack((np.zeros((m - 1, 1)), np.eye((m - 1))))
A = lambda x: np.vstack((a, np.array(([x**2]), dtype=float)))
B = lambda x: np.vstack((np.zeros((m - 1, 1)), np.array(([0.0]), dtype=float)))
x = x0
for i in range(n_rk04):
    dy0 = A(x) @ y0 + B(x)
    x += h / 2
    y1 = y0 + (h / 2) * dy0
    dy1 = A(x) @ y1 + B(x)
    y2 = y0 + (h / 2) * dy1
    dy2 = A(x) @ y2 + B(x)
    x += h / 2
    y3 = y0 + h * dy2
    dy3 = A(x) @ y3 + B(x)
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
    dy1 = A(x1) @ y1 + B(
        x1
    )  # Evaluación de derivadas en cada punto obtenido (x1,x2...)
    dy2 = A(x2) @ y2 + B(x2)
    dy3 = A(x3) @ y3 + B(x3)
    p = y0 + (4 / 3) * h * (2 * dy1 - dy2 + 2 * dy3)
    x4 = x3 + h
    dy4 = A(x4) @ p + B(x4)

    y = y2 + (h / 3) * (
        dy2 + 4 * dy3 + dy4
    )  # Linea principal para diferenciar a Milne con Hamming

    y0 = y1.copy()
    y1 = y2.copy()
    y2 = y3.copy()
    y3 = y.copy()
    x1 = x2
    x2 = x3
    x3 = x4
    resultados[4 + i, :] = np.concatenate(([4 + i, x3], y3.flatten()))

print(resultados)
