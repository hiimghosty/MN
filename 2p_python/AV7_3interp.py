# Es una verga hacer por interpolacion, no recomiendo

import numpy as np

t = np.array([0.02, 0.0204, 0.0208, 0.0212, 0.216], dtype=float)
f = np.array([34.949526, 34.951115, 34.952655, 34.954146, 34.95559], dtype=float)
x = t
y = f
n = len(x)
P = np.poly1d([0.0])

################RUTINA012########################## #LAGRANGE
for i in range(n):
    a = np.delete(np.arange(n), i)
    p = np.poly1d([1, -x[a[0]]])
    for j in range(1, n - 1):
        p = np.polymul(p, np.poly1d([1, -x[a[j]]]))
    P += y[i] * p / p(x[i])


print(P)

f = P
t = 0.0208
h = np.abs(x[0] - x[1])

d1y = (f(t + h) - f(t)) / h
