import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import grad


def M1(t):
    return (t - 2) * jnp.sqrt(t + 0.01) - np.exp(4 * t - 4) + t + 200


X = jnp.linspace(0, 10, 100)
plt.grid
plt.plot(X, M1(X))
plt.show()


def M2(t):
    return -(1) / (1 + t**2)


n = 100
tol = 1e-5
p0 = 0.5
f = M1
df = grad(f)


################RUTINA006########################## #NEWTON RHAPSON
for i in range(n):
    p = p0 - f(p0) / df(p0)
    err = jnp.abs(p - p0)
    relerr = jnp.abs(err / p)
    if tol > err or tol > relerr or tol > jnp.abs(f(p)):
        break
    p0 = p


print(p)
