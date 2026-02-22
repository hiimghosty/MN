# Ejercicio de la concentración del medicamento, usando Newton-Raphson
import jax.numpy as jnp
from jax import grad

# c(t) = A*t*e^(-t/3)
# La concentración máxima ocurre cuando c'(t) = 0

concentracionMaxima = 1.5  # mg/mL


# Derivada simplificada de c(t):
# c'(t) = A*e^(-t/3)*(1 - t/3)
# Para el máximo: 1 - t/3 = 0
def g(x):
    return 1 - x / 3


dg = grad(g)

tol = 1e-3
p0 = 1.0
n = 100

for i in range(n):
    p = p0 - g(p0) / dg(p0)
    err = jnp.abs(p - p0)
    relerr = jnp.abs(err / p)
    if tol > err or tol > relerr or tol > jnp.abs(g(p)):
        break
    else:
        p0 = p

print("\nb) Tiempo en que se llega a la concentración máxima (h):", f"{float(p):0.6f}")

# Con ese tiempo hallamos A:
# 1.5 = A * p * e^(-p/3)
A = concentracionMaxima / (p * jnp.exp(-p / 3))
print("a) Cantidad A a inyectar (mg):", f"{float(A):0.6f}")


# -------------------------
# PARTE 2
# -------------------------

# c) Cantidad adicional
# Cuando la concentración cae a 0.3 mg/mL,
# necesitamos que la nueva dosis aporte:
# 1.5 - 0.3 = 1.2 mg/mL en su máximo

concentracion_caida = 0.3
concentracion_faltante = concentracionMaxima - concentracion_caida

# El máximo de una dosis ocurre en t=3
# y vale: A_extra * 3 * e^{-1}
A_extra = concentracion_faltante / (3 * jnp.exp(-1))

print("c) Cantidad adicional a inyectar (mg):", f"{float(A_extra):0.6f}")


# d) Instante cuando la concentración cae a 0.3 mg/mL
# Resolver: A*t*e^(-t/3) - 0.3 = 0
# (rama descendente → p0=10)


def h(x):
    return A * x * jnp.exp(-x / 3) - concentracion_caida


dh = grad(h)
p0_2 = 10.0

for i in range(n):
    p2 = p0_2 - h(p0_2) / dh(p0_2)
    err2 = jnp.abs(p2 - p0_2)
    relerr2 = jnp.abs(err2 / p2)
    if tol > err2 or tol > relerr2 or tol > jnp.abs(h(p2)):
        break
    else:
        p0_2 = p2

print("d) Instante para aplicar la segunda inyección (h):", f"{float(p2):0.6f}")
