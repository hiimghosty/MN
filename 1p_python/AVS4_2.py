import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

x_dat = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
y_dat = jnp.array([0.80, 0.92, 1.24, 1.40, 1.36, 1.42, 1.60, 1.67, 1.54], dtype=float)


def f(var):
    A, B = var
    model = (x_dat) / (A * x_dat + B)
    return jnp.sum((model - y_dat) ** 2)


sistema = jax.grad(f)
jacob_sist = jax.hessian(f)
m = 100
P0 = jnp.array([0.5, 0.5], dtype=float)
tol = 1e-3

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


y_pred = x_dat / (x_dat * P[0] + P[1])
SS_tot = jnp.sum((y_dat - jnp.mean(y_dat)) ** 2)
SS_res = jnp.sum((y_dat - y_pred) ** 2)
R2 = 1 - (SS_res / SS_tot)

print("Los coeficientes del modelo son ", P)
print("El coeficiente de determinacion es: ", R2)
print("Iteraciones", i + 1)
print(f"Error = {err:0.6f}")
print(f"Error RELATIVO = {relerr:0.6f}")
