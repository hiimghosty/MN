import jax.numpy as jnp
import numpy as np
from jax import grad

y_dat = jnp.array(
    [999.839, 999.898, 999.940, 999.964, 999.972, 999.964, 999.940, 999.901],
    dtype=float,
)

x_dat = jnp.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=float)

x = x_dat
y = y_dat
m = len(x)

################RUTINA011##########################
v = np.vander(x, N=m, increasing=True)
a = np.linalg.solve(v, y)

print("Coeficientes del polinomio:")
print(a)

# ejemplo de evaluación

xp = 2.5  # ME PIDE EVALUAR PARA T = 2.5
yp = 0.0

for i in range(m):
    yp += (
        a[i] * xp**i
    )  # ESTO SIGNIFICA, AGARRAR EL COEFICIENTE i E DE a, y multiplicar por xp^i, desde i = 0 hasta i = 7

print("P(2.5) =", yp)


def f(t):
    s = 0.0
    for i in range(1, m):
        s += i * a[i] * t ** (i - 1)

    return s


df = grad(f)
p0 = 4.0
n = 20
tol = 1e-2

################RUTINA006########################## #NEWTON RHAPSON
for i in range(n):
    p = p0 - f(p0) / df(p0)
    err = jnp.abs(p - p0)
    relerr = jnp.abs(err / p)
    if tol > err or tol > relerr or tol > jnp.abs(f(p)):
        break
    p0 = p


print(f"c) Para densidad maxima la temperatura es: {p:0.6f}")
