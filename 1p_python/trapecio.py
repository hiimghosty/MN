# trapecio

import jax.numpy as jnp


def f(var):
    w, d, th, la = var
    F1 = w - (2 * d / jnp.tan(th)) + (2 * d / jnp.sin(th))
    F2 = d * (w - d / jnp.tan(th))
    return F1 - la * (F2 - 50)
