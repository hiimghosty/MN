import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, hessian

jax.config.update("jax_enable_x64", True)

c = np.array([0.5, 0.8, 1.5, 2.5, 4], dtype=float)
k = np.array([0.7, 1.6, 3.9, 6.1, 7.6], dtype=float)


def f(var):
    # var = coeficientes del modelo
    # ejemplo: var = [a, b]
    kmax, cs = var
    model = (kmax * (c) ** 2) / (cs + (c) ** 2)  # ejemplo de modelo

    return jnp.sum((model - k) ** 2)


sistema = grad(f)
jacob_sist = hessian(f)


################RUTINA009-Newton Rapson sistemas no lineales(multidimensional)##########################
#
# Qué hace: Resuelve un sistema de ecuaciones (sea el S.E lineal o no).
# sistema(): Es el sistema de ecuación que se quiere resolver.
# jacob_sist(): Es el jacobiano del sistema de ecuación que se quiere resolver.
# P0: Punto de partida (debe ser condicionado en el ejercicio).
# m: Cantidad máxima de iteraciones.
m = 20
P0 = jnp.array([8, 3], dtype=float)  # NO era dato pero tiene que ser
tol = 1e-12
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
kmax = P[0]
cs = P[1]


################RUTINA010-Coef. de correlacion##########################
def R2(y_data, y_pred):
    y_mean = np.mean(y_data)
    SS_res = np.sum((y_data - y_pred) ** 2)
    SS_tot = np.sum((y_data - y_mean) ** 2)
    return 1 - SS_res / SS_tot


y_pred = (kmax * (c) ** 2) / (cs + (c) ** 2)  # ejemplo de modelo
print(y_pred)
R2 = R2(k, y_pred)
print(f"Valor de R2 {R2:0.6f}")
estimacionK = (kmax * (2) ** 2) / (cs + (2) ** 2)  # ejemplo de modelo
print(estimacionK)
