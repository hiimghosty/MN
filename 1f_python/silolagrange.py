import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, hessian

np.set_printoptions(suppress=True)
jax.config.update("jax_enable_x64", True)


def AreaCilindro(r, h):

    return (np.pi) * r * h * 2


def AreaSemiEsfera(r):
    return 2 * (np.pi) * (jnp.pow(r, 2))


def AreaBase(r):
    return (np.pi) * (jnp.pow(r, 2))


costoCilindro = 10  # usd
costoSemiesfera = 25
costoBase = 30
volumenMaximo = 1000  # m^3


def Volumentotal(r, h):
    return np.pi * jnp.pow(r, 2) * h + 0.5 * (4 / 3) * np.pi * jnp.pow(r, 3)


def lagrange(var):
    r, h, landa = var
    return (
        costoCilindro * AreaCilindro(r, h)
        + costoBase * AreaBase(r)
        + costoSemiesfera * AreaSemiEsfera(r)
        + landa * (Volumentotal(r, h) - volumenMaximo)
    )


def costoTotal(r, h):
    return (
        costoCilindro * AreaCilindro(r, h)
        + costoBase * AreaBase(r)
        + costoSemiesfera * AreaSemiEsfera(r)
    )


sistema = grad(lagrange)
jacob_sist = hessian(lagrange)
###################################################
################RUTINA009-Newton Rapson sistemas no lineales(multidimensional)##########################
#
tol = 1e-1
m = 20
P0 = np.array(
    [4, 15, -5], dtype=float
)  # el punto inicial que da en el TAA es terrible, ni en pedo converge asi
# rutina:
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

print(P)
costo = costoTotal(P[0], P[1])
print(f"El radio es: {P[0]:.6f}")
print(f"La altura es: {P[1]:.6f}")
print(f"El costo es: {costo:.6f}")
