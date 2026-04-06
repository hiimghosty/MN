# In[1]:
import numpy as np

x = np.array([33.2, 44.3, 50, 55.6, 61])
y = np.array([11, 17, 16.6, 16.9, 16])


def newton_interp(x, y):
    n = len(x)

    P = np.poly1d([y[0]])
    p = np.poly1d([1, -x[0]])

    for i in range(1, n):
        a = (y[i] - P(x[i])) / p(x[i])
        P += a * p
        p = np.polymul(p, np.poly1d([1, -x[i]]))

    return P


def regulafalsi(a, b, f):
    tol = 1e-8
    n = 50
    c0 = a
    for i in range(n):
        c = (b * f(a) - a * f(b)) / (f(a) - f(b))
        err = np.abs(c0 - c)
        relerr = np.abs(err / c)
        if tol > err or tol > relerr or tol > np.abs(f(c)):
            break
        else:
            if f(a) * f(c) < 0:
                b = c
            else:
                a = c
    c0 = c
    return c


v = newton_interp(x, y)
print(v)
print(v(35.5))

f = v - 14.88
p2 = regulafalsi(30, 45, f)
print(p2)

dv = np.polyder(v)
p3 = regulafalsi(30, 45, dv)
print(p3)
p4 = v(p3)
print(p4)


x = np.array([33.2, 44.3, 50, 55.6, 61, 77.7])
y = np.array([11, 17, 16.6, 16.9, 16, 18.3])

v2 = newton_interp(x, y)

print(v2(72))
