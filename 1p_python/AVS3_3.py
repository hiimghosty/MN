# Ejercicio del cono maembo
# COMENTARIOS IMPORTANTES: A la fecha de hoy (1/03/2026) solo me coincide el valor de r, el resto de valores me difieren por
# 0.000004 luego de revisar mucho, llegue a la conclusion de que esto sucede porque para la respuesta de r usaron el vector P
# Que hallamos con newton rhapson, pero por algun motivo para hallar el resto de valores despejaron de las formulas y redondearon
# Parcialmente, y luego de vuelta al final, contradiciendo totalmente el enunciado
# Haciendo que, llegar a la respuesta verdadera sea basicamente jugar a las adivinanzas
import jax
import jax.numpy as jnp
from jax import grad, jacfwd

jax.config.update("jax_enable_x64", True)


def V(r, h):
    return 1 / 3 * (jnp.pi * (r**2) * h)


def areaLateral(r, h):
    return jnp.pi * r * jnp.sqrt(pow(r, 2) + pow(h, 2))


def areaBase(r):
    return jnp.pi * pow(r, 2)


def L(var):
    r, h, lm = var
    return 25 * areaLateral(r, h) + 50 * areaBase(r) + lm * (V(r, h) - 50)


# Sistema de ec. no lineales

sistema = grad(L)

# Parametros
P0 = jnp.array(
    [1.0, 47.746483, -75.0], dtype=float
)  # El enunciado solo habla de r=1, despejando podemos intentar usar un vector aproximado a la solucion
tol = 1e-3
m = 20
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
C_res = 100 * V(dim_r, dim_h) + 25 * areaLateral(dim_r, dim_h) + 50 * areaBase(dim_r)

print("------------------")
print(f"a) Radio del silo: {dim_r:0.6f}")
print(f"b) Altura del silo: {dim_h:0.6f} ")
print(f"d) Costo del silo: {C_res:0.6f}")
print(f"Valor de lambda {P[2]:0.6f}")
