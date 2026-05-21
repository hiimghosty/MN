# ejercicio del tanque
import numpy as np

y0 = np.array([0.0])
x0 = 0.0
h = 1
n_rk04 = 3  # puntos iniciales
n_milne = 7  # en total necesito 10 puntos, xq me pide para t = 10 en uno de los items
################RUTINA023##########################
m = len(y0)
resultados = np.zeros(
    (n_rk04 + n_milne + 1, 2 + m)
)  # una fila extra para las condiciones inciales, y 2 columnas extra , uno para i y uno para x
resultados[0, :] = np.concatenate(
    ([0, x0], y0.flatten())
)  # cargar valores iniciales, 0 por la iteracion 0
a = np.hstack((np.zeros((m - 1, 1)), np.eye((m - 1))))
A = lambda x: np.vstack((a, np.array(([0.0]), dtype=float)))
B = lambda x: np.vstack(
    (
        np.zeros((m - 1, 1)),
        np.array(
            ([-(1600 / (np.pi * 40**2)) + (4800 / (np.pi * 40**2)) * np.sin(x) ** 2]),
            dtype=float,
        ),
    )
)
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
    resultados[1 + i, :] = np.concatenate(([1 + i, x], y.flatten()))
    # ese +1 es dedbido a que nos queremos saltar la primera fila, ya que no queremos remplazar valores iniciales
    # agarramos todas las columnas
    # ahora, el tema de concatenar es concatenar la cantidad de iteraciones, valor actual de x, e y, la y le hacemos flatten xq es un array

# ahora, ya que tenemos los valores de rk04, pasamos a milne
################RUTINA024##########################

x0, y0 = resultados[0, 1], resultados[0, 2:].copy()
x1, y1 = resultados[1, 1], resultados[1, 2:].copy()
x2, y2 = resultados[2, 1], resultados[2, 2:].copy()
x3, y3 = resultados[3, 1], resultados[3, 2:].copy()
for i in range(n_milne):
    dy1 = A(x1) @ y1 + B(x1)
    dy2 = A(x2) @ y2 + B(x2)
    dy3 = A(x3) @ y3 + B(x3)
    p = y0 + (4 / 3) * h * (2 * dy1 - dy2 + 2 * dy3)
    if i > 0:
        mod = p + (28 / 29) * (y3 - p0)
    else:
        mod = p
    x4 = x3 + h
    dy4 = A(x4) @ mod + B(x4)
    y = y2 + (h / 3) * (dy2 + 4 * dy3 + dy4)
    y0 = y1.copy()
    y1 = y2.copy()
    y2 = y3.copy()
    y3 = y.copy()
    x1 = x2
    x2 = x3
    x3 = x4
    p0 = p.copy()

    resultados[4 + i, :] = np.concatenate(([4 + i, x4], y.flatten()))

print(resultados)
