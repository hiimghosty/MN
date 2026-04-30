import numpy as np

x = np.array([0, 2, 4], dtype=float)
y = np.array([0, 10, 15], dtype=float)

n = len(x)

P = np.poly1d([0])
################RUTINA012########################## #LAGRANGE
for i in range(n):
    a = np.delete(np.arange(n), i)
    p = np.poly1d([1, -x[a[0]]])
    for j in range(1, n - 1):
        p = np.polymul(p, np.poly1d([1, -x[a[j]]]))
    P += y[i] * p / p(x[i])

print(P)

a = 0
b = 4
f = P
n = 15
################RUTINA015########################## # SIMPSON 1/3
A = 0.0
h = (b - a) / n
for xi in np.arange(a, b, 2 * h):
    A = A + (h / 3) * (f(xi) + 4 * f(xi + h) + f(xi + 2 * h))
print(f"Volumen {A:0.6f}")
