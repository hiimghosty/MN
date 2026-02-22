# EJercicio ulises, cable colgando
import jax.numpy as jnp


def f(x):
    return (1 / x) * (jnp.cosh((x * L) / 2) - 1) - 15


df = jax.grad(f)

p0 = 1
tol = 1e-3
n = 100
