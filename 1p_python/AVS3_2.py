# El ejercicio ese del silo
import jax.numpy as jnp
from jax import grad, jacfwd

# Datos del problema
costo_Semiesfera = 24  # USD
costo_ParedCilindrica = 14
costo_Base = 35
volumenFijo = 1000  # m3


# Funcion a optimizar L (lambda,r,h) = C(r,h) + lambda * ( V(r,h) - 1000)
def C(r, h):  # NO use las constantes de costo xq quedaba muy largo
    return 28 * jnp.pi * r * h + 83 * jnp.pi * jnp.pow(r, 2)


def V(r, h):
    return jnp.pi * jnp.pow(r, 2) * h + 2 / 3 * jnp.pi * jnp.pow(r, 3)


# Multiplicador de lagrange


def L(var):
    r, h, lm = var
    return C(r, h) + lm * (1000 - V(r, h))


# Sistema de ec. no lineales

sistema = grad(L)

# Parametros
P0 = jnp.array([10, 10, 10], dtype=float)  # DATO DEL PROBLEMA
tol = 1e-3
m = 50
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


print("Analsis de convergencia:")
print("Error absoluto: ", err)
print("Error relativo: ", relerr)
print("Cantidad de iteraciones: ", i + 1)
print(
    "Valor de |F(P)|: ", F_norm
)  # Como estamos hallando raices de L(lambda,r,h) mientras mas cerca de 0 este, mejor

dim_r = P[0]
dim_h = P[1]
L_res = jnp.abs(L(P))
C_res = C(dim_r, dim_h)

print("------------------")
print(f"a) Radio del silo: {dim_r:0.6f}")
print(f"b) Altura del silo: {dim_h:0.6f} ")
print(
    f"c) Valor del multiplicador de Lagrange:  {L_res:0.6f} "
)  # realmente este es el valor de la funcion lagraniana
print(f"d) Costo del silo: {C_res:0.6f}")
print(f"Valor de lambda {P[2]:0.6f}")
