import jax.numpy as jnp
from jax import grad


def M1(t):
    return (t - 2) * jnp.sqrt(t + 0.01) - jnp.exp(4 * (t - 1)) + t + 200


def M2(t):
    return (1 / ((t**2) + 2)) + jnp.cos(3 * jnp.acos((t - 1) / 2)) + t + 200


def newtonRhapson(n, tol, p0, f):
    df = grad(f)
    ################RUTINA006########################## #NEWTON RHAPSON
    for i in range(n):
        p = p0 - f(p0) / df(p0)
        err = jnp.abs(p - p0)
        relerr = jnp.abs(err / p)
        if tol > err or tol > relerr or tol > jnp.abs(f(p)):
            break
        p0 = p  # En el codigo de ulises idento mal este
    return p


n = 100
tol = 1e-5
p0 = 3.0
f = grad(M1)

t1optimo = newtonRhapson(n, tol, p0, f)
p0 = 0.4
g = grad(M2)
t2optimo = newtonRhapson(n, tol, p0, g)
print(
    f"El tiempo optimo de la primera maquina es: {t1optimo:0.6f} y su produccion es {M1(t1optimo):0.6f}"
)
print(
    f"El tiempo optimo de la segunda maquina es: {t2optimo:0.6f} y su produccion es {M2(t2optimo):0.6f}"
)


# Como la segunda maquina produce mas en menos tiempo, fijamos ese tiempo
# Ahora, debemos hallar un tiempo t1 tal que M1(t) + M2(t2optimo) = 400
def F(x):
    return M1(x) + M2(t2optimo) - 400


# Usamos newton rhapson para hallar ese valor
p0 = 3.0
f = F
t1optimo = newtonRhapson(n, tol, p0, f)
print(
    f"El tiempo t1 real es: {t1optimo:0.6f} y la produccion es {F(t1optimo) + 400:0.6f}"
)
