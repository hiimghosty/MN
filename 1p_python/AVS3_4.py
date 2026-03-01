# Ejercicio del saltelite
import jax
import jax.numpy as jnp
from jax import jacfwd

jax.config.update("jax_enable_x64", True)
# Constantes
G = 6.674e-11
M = 5.972e24
r = 7e6


def sistema(var):
    x, y, vx, vy = var
    f1 = x**2 + y**2 - (r + 1300) ** 2
    f2 = (G * M * x) / ((x**2 + y**2) ** (3 / 2)) + (vx + 2)
    f3 = (G * M * y) / ((x**2 + y**2) ** (3 / 2)) + (vy + 4)
    f4 = vx - 0.5 * vy
    return jnp.array([f1, f2, f3, f4], dtype=float)


# Parametros iniciales

P0 = jnp.array([r, r, 0, 0], dtype=float)
m = 20
tol = 1e-9
jacob_sist = jacfwd(sistema)


################RUTINA008########################## #NEWTON RHAPSON MULTI DIMENSIONAL
for i in range(m):
    F = sistema(P0)
    J = jacob_sist(P0)
    deltaP = jnp.linalg.solve(J, -F)
    P = P0 + deltaP
    err = jnp.linalg.norm(P - P0)
    relerr = err / jnp.linalg.norm(P)
    F_norm = jnp.linalg.norm(sistema(P))
    if tol > err or tol > relerr or tol > F_norm:
        break
    else:
        P0 = P.copy()


print(f"x = {float(P[0]):.6f}")  # EL ENUNCIADO NO PIDE ASI, PIDE EN X*10^6
print(f"y = {float(P[1]):.6f}")  # EL ENUNCIADO NO PIDE ASI, PIDE EN Y*10^6
print(f"vx = {float(P[2]):.6f}")
print(f"vy = {float(P[3]):.6f}")
print(f"Error = {err:0.6f}")
print(f"Rapidez: {jnp.sqrt(P[2] ** 2 + P[3] ** 2):0.6f}")
