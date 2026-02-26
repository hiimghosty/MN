# Clase con Ulises
# x^2 + y^2 = 3
import jax.numpy as jnp


def sistema(var):
    x, y = var
    f1 = x**2 + y**2 - 3
    f2 = x * y - 1
    return jnp.array([f1, f2], dtype=float)


p0 = jnp.array([1.0, 0.0])
m = 100
tol = 1e-3
