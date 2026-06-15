import numpy as np

np.set_printoptions(suppress=True)

# datos
Pesocuerpo = 4
g = 10  # m/s^2
y0 = np.array([[1], [-8]], dtype=float)
h = 0.1
n_rk04 = 3
n_milne = 80
x0 = 0
Kresorte = 2

################RUTINA023-Runge-kutta (Rk4)##########################
m = len(y0)
resultados = np.zeros((n_rk04 + n_milne + 1, 2 + m))
resultados[0, :] = np.concatenate(([0, x0], y0.flatten()))
a = np.hstack((np.zeros((m - 1, 1)), np.eye((m - 1))))
A = lambda x: np.vstack((a, np.array(([-Kresorte * g / 4, -g / 4]), dtype=float)))
B = lambda x: np.vstack((np.zeros((m - 1, 1)), np.array(([-g]), dtype=float)))
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


# AHOra hamming modificado
################RUTINA027-Hamming modificado.##########################
# rutina
x0, y0 = resultados[0, 1], resultados[0, 2:].reshape(-1, 1).copy()
x1, y1 = resultados[1, 1], resultados[1, 2:].reshape(-1, 1).copy()
x2, y2 = resultados[2, 1], resultados[2, 2:].reshape(-1, 1).copy()
x3, y3 = resultados[3, 1], resultados[3, 2:].reshape(-1, 1).copy()
m = len(y0)
a = np.hstack((np.zeros((m - 1, 1)), np.eye((m - 1))))
x1 = x0 + h
x2 = x1 + h
x3 = x2 + h
R = y0.copy()
R = np.hstack((R, y1))
R = np.hstack((R, y2))
R = np.hstack((R, y3))
for i in range(n_milne):
    dy1 = A(x1) @ y1 + B(x1)
    dy2 = A(x2) @ y2 + B(x2)
    dy3 = A(x3) @ y3 + B(x3)
    p = y0 + (4 / 3) * h * (2 * dy1 - dy2 + 2 * dy3)
    if i > 0:
        mod = p + (112 / 121) * (y3 - p0)
    else:
        mod = p
    x4 = x3 + h
    dy4 = A(x4) @ mod + B(x4)
    y = (9 * y3 - y1) / 8 + (3 / 8) * h * (-dy2 + 2 * dy3 + dy4)
    y0 = y1.copy()
    y1 = y2.copy()
    y2 = y3.copy()
    y3 = y.copy()
    x1 = x2
    x2 = x3
    x3 = x4
    p0 = p.copy()
    resultados[n_rk04 + i + 1, :] = np.concatenate(([n_rk04 + i + 1, x4], y.flatten()))


print(resultados)
# ---------------------------------------------------------------------------
# INTERPRETACION FISICA (resorte amortiguado: y=posicion, y'=velocidad)
#
# Posicion de equilibrio  -> donde la masa quedaria en reposo: y'=0 y y''=0.
#                            Da y_eq = -4/k = -2. La masa oscila alrededor de
#                            este valor y, al amortiguarse, termina ahi.
#
# Cruza el equilibrio      -> instante en que la POSICION vale y_eq (col y = -2).
#                            Ahi la masa va a su maxima rapidez (no se detiene).
#
# Desplazamiento extremo   -> punto mas lejano del equilibrio (mas alto o mas
#                            bajo). La masa se DETIENE un instante: y'=0
#                            (la VELOCIDAD cambia de signo).
#
# Posicion en el extremo   -> valor de y en ese instante; el signo indica el
#                            lado (y<0 -> por debajo del equilibrio).
# ---------------------------------------------------------------------------
