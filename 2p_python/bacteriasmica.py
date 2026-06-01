import numpy as np

# constantes
b0 = 0.029
k = 1.4 * 10 ** (-7)
h = 1
# valores iniciales
x0 = 0
y0 = np.array([[50976.0]])  # vector columna


# indice de reproduccion
def b(t):
    return b0 * (2 + np.sin(np.pi * t / 3))


# resultados generales
x_gen = [x0]
y_gen = [y0[0][0]]
dy_gen = []
# resultados de heun
x_heun = []
y_heun = []
################RUTINA022########################## HEUN
m = len(y0)
a = np.hstack((np.zeros((m - 1, 1)), np.eye((m - 1))))
A = (
    lambda x: np.vstack((a, np.array([b(x)], dtype=float)))
)  # Acá guardamos lo que está multiplicado por y o alguna de sus derivadas, en orden creciente de derivas (mismo orden en el que inicializaste tu punto inicial)
B = lambda x, y: np.vstack(
    (np.zeros((m - 1, 1)), np.array(([-k * y[0][0] ** 2]), dtype=float))
)
x = x0
n_heun = 3  # ?
for i in range(n_heun):
    dy1 = A(x) @ y0 + B(x, y0)
    dy_gen.append(
        dy1[0][0]
    )  # desde la derivada del punto inicial hasta -1 del punto final

    y1 = y0 + h * dy1
    x += h
    dy2 = A(x) @ y1 + B(x, y1)
    y = y0 + (h / 2) * (dy1 + dy2)
    y0 = y.copy()
    x_heun.append(x)
    y_heun.append(y)
    x_gen.append(x)
    y_gen.append(y[0][0])

dy1 = A(x) @ y0 + B(x, y0)
dy_gen.append(dy1[0][0])  # derivada del punto final
# print("las x de heun:",x_heun,"las y de heun:", y_heun, "derivadas:",dy_gen)

############### HAMMING
# valores iniciales
x0 = 0
y0 = np.array([50976.0])  # vector columna
y1 = y_heun[0]
y2 = y_heun[1]
y3 = y_heun[2]
################RUTINA026###########
m = len(y0)
a = np.hstack((np.zeros((m - 1, 1)), np.eye((m - 1))))
A = (
    lambda x: np.vstack((a, np.array([b(x)], dtype=float)))
)  # Acá guardamos lo que está multiplicado por y o alguna de sus derivadas, en orden creciente de derivas (mismo orden en el que inicializaste tu punto inicial)
B = lambda x, y: np.vstack(
    (np.zeros((m - 1, 1)), np.array(([-k * y[0][0] ** 2]), dtype=float))
)
x1 = x0 + h
x2 = x1 + h
x3 = x2 + h
n_hamm = 5  # tanteo nm
for i in range(n_hamm):
    dy1 = A(x1) @ y1 + B(x1, y1)
    dy2 = A(x2) @ y2 + B(x2, y2)
    dy3 = A(x3) @ y3 + B(x3, y3)
    p = y0 + (4 / 3) * h * (2 * dy1 - dy2 + 2 * dy3)
    x4 = x3 + h
    dy4 = A(x4) @ p + B(x4, p)
    y = (9 * y3 - y1) / 8 + (3 / 8) * h * (-dy2 + 2 * dy3 + dy4)
    y0 = y1.copy()
    y1 = y2.copy()
    y2 = y3.copy()
    y3 = y.copy()
    x1 = x2
    x2 = x3
    x3 = x4
    y_gen.append(y[0][0])
    x_gen.append(x4)  # ?
    dy1 = A(x4) @ y + B(x4, y)
    dy_gen.append(dy1[0][0])  # derivada del punto final

# print("las x:",x_gen,"las y:", y_gen)

print(f"a) el nro de bacterias a los 2 seg {y_gen[2]:.6f}")
print(f"b) el nro de bacterias a los 6 seg {y_gen[6]:.6f}")
print(f"c) el indice de crecimiento a los 3 seg {dy_gen[3]:.6f}")
print(f"d) el indice de crecimiento a los 5 seg {dy_gen[5]:.6f}")

x = x_gen
y = y_gen
n = len(x)
# interpolacion usando lagrange
p = np.poly1d([1, -x[0]])
P = np.poly1d(y[0])
################RUTINA013########################## LAGRANGE
for i in range(1, n):
    a = (y[i] - P(x[i])) / p(x[i])
    P += a * p
    p = np.polymul(p, np.poly1d([1, -x[i]]))

print(f"e) estimacion para t=4.5 usando el polinomio: {P(4.5):.6f}")
