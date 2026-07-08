################LIBRERIAS##########################
import numpy as pomelo
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", False)
from jax import jacfwd, grad

A = pomelo.array(1.0)
print(A)

b = (5**2)*3
b = (pomelo.pow(5, 2)) * 3

