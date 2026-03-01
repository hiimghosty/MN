# no coincide  nada, revisar
import jax
import jax.numpy as jnp
from jax import jacfwd

jax.config.update("jax_enable_x64", True)

A_obj = 50.0


# Reparametrización:
# d = exp(u) > 0
# b = exp(s) > 0
# w = b + 2d  (talud 1:1)
def d_from(u):
    return jnp.exp(u)


def b_from(s):
    return jnp.exp(s)


def w_from(u, s):
    d = d_from(u)
    b = b_from(s)
    return b + 2.0 * d


def area_from(u, s):
    d = d_from(u)
    b = b_from(s)
    # A = d * ( (w+b)/2 ) con w=b+2d -> (w+b)/2 = b + d
    return d * (b + d)


def perimetro_from(u, s):
    d = d_from(u)
    b = b_from(s)
    # P = b + 2*sqrt(2)*d
    return b + 2.0 * jnp.sqrt(2.0) * d


# Lagrangiana L = P + λ(A - A_obj)
def lagrangiana(var):
    u, s, lam = var
    return perimetro_from(u, s) + lam * (area_from(u, s) - A_obj)


gradL = jax.grad(lagrangiana)


def sistema(var):
    # Sistema KKT: [∂L/∂u, ∂L/∂s, ∂L/∂λ] = 0
    return jnp.array(gradL(var), dtype=float)


# Parámetros
m = 50
tol = 1e-9
jacob_sist = jacfwd(sistema)

# Chute inicial:
# El enunciado dice "variables en 3" y λ=0.1.
# Como nuestras variables libres son u y s, elegimos d=3 y b=3:
# u0 = ln(3), s0 = ln(3)
P0 = jnp.array([jnp.log(3.0), jnp.log(3.0), 0.1], dtype=float)

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

# Recuperar variables físicas
u_opt, s_opt, lam_opt = P
d_opt = float(d_from(u_opt))
b_opt = float(b_from(s_opt))
w_opt = float(w_from(u_opt, s_opt))

P_min = float(perimetro_from(u_opt, s_opt))
grad_norm = float(jnp.linalg.norm(sistema(P)))
A_val = float(area_from(u_opt, s_opt))

print("RESULTADOS (6 decimales, sin notación exponencial)")
print(f"w (base mayor) = {w_opt:.6f}  [m]")
print(f"d (altura)     = {d_opt:.6f}  [m]")
print(f"|lambda|       = {abs(float(lam_opt)):.6f}")
print(f"||grad L||     = {grad_norm:.6f}")
print(f"P_min          = {P_min:.6f}  [m]")
print(f"Iteraciones i  = {i + 1}")
print(f"(chequeo) b    = {b_opt:.6f}  [m]")
print(f"(chequeo) Area = {A_val:.6f}  [m^2]")
