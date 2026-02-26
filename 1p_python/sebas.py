import numpy as np

# La funcion escrita es en funcion a (w,t) f(w) = (g/(2.0*(w)**2))*((((np.exp(w*t))-(np.exp(-w*t)))/(2.0))-(jnp.sin(w*t)))
g = 32.17
t = 2.4
x_val = 1.3


def f(w):
    return (g / (2.0 * (w) ** 2)) * (
        (((np.exp(w * t)) - (np.exp(-w * t))) / (2.0)) - (np.sin(w * t))
    ) - x_val


tol = 1e-3
n = 50
x0 = -2
x1 = -1
x2 = x0
for i in range(n):
    x2 = (x1 * f(x0) - x0 * f(x1)) / (f(x0) - f(x1))
    err = np.abs(x2 - x1)
    relerr = np.abs(err / x2)
    if tol > err or tol > relerr or tol > np.abs(f(x2)):
        break
    x0 = x1
    x1 = x2

print(f"El valor de w es {x2:0.6f}")
print(f"error relativo de: {err:0.6f}")
print(f"Error absoluto de: {relerr:0.6f}")
print("El numero de iteraciones es:", i + 1)
