import numpy as np

y0 = np.array([3.0])
x0 = 1.0
n = 3
h = 0.1
n_milne = 2

################RUTINA023##########################
m = len(y0)
a = np.hstack((np.zeros((m - 1, 1)), np.eye((m - 1))))
A = lambda x, y: np.vstack((a, np.array([(x / y**2)], dtype=float)))
B = lambda x: np.vstack((np.zeros((m - 1, 1)), np.array([0.0], dtype=float)))

result = np.zeros((n + n_milne + 1, 2 + m))
result[0, :] = np.concatenate(([0], [x0], y0.flatten()))

x = x0
for i in range(n):
    dy0 = A(x, y0[0]) @ y0 + B(x)  # <-- agregado y0[0]
    x += h / 2
    y1 = y0 + (h / 2) * dy0
    dy1 = A(x, y1[0]) @ y1 + B(x)
    y2 = y0 + (h / 2) * dy1
    dy2 = A(x, y2[0]) @ y2 + B(x)
    x += h / 2
    y3 = y0 + h * dy2
    dy3 = A(x, y3[0]) @ y3 + B(x)
    y = y0 + (h / 6) * (dy0 + 2 * dy1 + 2 * dy2 + dy3)
    y0 = y.copy()
    result[1 + i, :] = np.concatenate(([1 + i], [x], y0.flatten()))  # <-- corchetes


################RUTINA024##########################

x1 = x0 + h
x2 = x1 + h
x3 = x2 + h

y0 = result[0, 2:].reshape(m, 1)
y1 = result[1, 2:].reshape(m, 1)
y2 = result[2, 2:].reshape(m, 1)
y3 = result[3, 2:].reshape(m, 1)

for i in range(n_milne):
    dy1 = A(x1, y1[0]) @ y1 + B(x1)
    dy2 = A(x2, y2[0]) @ y2 + B(x2)
    dy3 = A(x3, y3[0]) @ y3 + B(x3)

    p = y0 + (4 / 3) * h * (2 * dy1 - dy2 + 2 * dy3)
    x4 = x3 + h
    dy4 = A(x4, p[0]) @ p + B(x4)

    y = y2 + (h / 3) * (dy2 + 4 * dy3 + dy4)

    y0 = y1.copy()
    y1 = y2.copy()
    y2 = y3.copy()
    y3 = y.copy()
    x1 = x2
    x2 = x3
    x3 = x4
    result[4 + i, :] = np.concatenate(([4 + i], [x4], y0.flatten()))  # <-- corchetes
print(result)
# dp agrego las dy, estaba disociando asi q no anote
