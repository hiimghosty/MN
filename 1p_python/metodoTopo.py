import jax.numpy as jnp
import numpy as np

#   Declaracion de DATOS
x = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=float)
y = np.array(
    [999.839, 999.898, 999.940, 999.964, 999.972, 999.964, 999.940, 999.901],
    dtype=float,
)

# x = variables independientes
# y = variables dependientes

m = len(x)
a = np.poly1d([1, -x[0]])
P = np.poly1d([y[0]])


# RUTINA METODO DIRECTO
################RUTINA011##########################
v = np.vander(x, N=m, increasing=True)
a = np.linalg.solve(v, y)

print(a)

t = np.array([a[7], a[6], a[5], a[4], a[3], a[2], a[1], a[0]], dtype=float)
q = np.poly1d(t)

print(q)

#   ITEM a)
print("La densidad correspondiente a una temperatura de 2,4 es: ", np.around(q(2.4), 6))


#   ITEM b)
print("El error para una densidad de 999.952 es: ", 999.952 - q(2.4))


#   ITEM c)
p0 = 4.0
n = 50
tol = 1e-2


f = np.polyder(q)
df = np.polyder(f)
p0 = 4.0
tol = 1e-2
n = 50
################RUTINA006########################## #NEWTON RHAPSON
for i in range(n):
    p = p0 - f(p0) / df(p0)
    err = jnp.abs(p - p0)
    relerr = jnp.abs(err / p)
    if tol > err or tol > relerr or tol > jnp.abs(f(p)):
        break
    p0 = p


print(f"c) Para densidad maxima la temperatura es: {p:0.6f}")
