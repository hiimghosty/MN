import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
x_dat = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
y_dat = np.array([0.58, 0.37, 0.22, 0.29, 0.34, 0.12, 0.13, 0.23, 0.31], dtype=float)


def f(var):
    A, B = var
    model = B / (x_dat + A)
    return np.sum((model - y_dat) ** 2)


m = 15
tol = 1e-8
P0 = np.array([1, 1], dtype=float)
sistema = jax.grad(f)
jacob_sist = jax.hessian(f)
################RUTINA009##########################
for i in range(m):
    F = sistema(P0)
    J = jacob_sist(P0)
    deltaP = np.linalg.solve(J, -F)
    P = P0 + deltaP
    err = np.linalg.norm(P - P0)
    relerr = err / np.linalg.norm(P)
    F_norm = np.linalg.norm(sistema(P))
    if tol > err or tol > relerr or tol > F_norm:
        break
    else:
        P0 = P.copy()

y_pred = P[1] / (x_dat + P[0])
SST = jnp.sum((y_dat - jnp.mean(y_dat)) ** 2)
SSR = jnp.sum((y_dat - y_pred) ** 2)

R = 1 - SSR / SST

print("A: ", P[0])
print("B: ", P[1])
print("R: ", R)
