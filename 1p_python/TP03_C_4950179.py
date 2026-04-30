import jax.numpy as jnp
from jax import jacfwd

V_max = 25
costoArea = 70
costoVol = 22
costoFleteVol = 30
radioini=4
lmini=3
# opc 2


def sistema(var):

    r,Lamb=var

    ec1= -12.566370614359*Lamb*r**2 + 1759.2918860103*r

    ec2=25 - 4.1887902047864*r**3

    return jnp.array(([ec1,ec2]),dtype=float)

def Jacob(var):

    r,Lamb=var

    a11=-25.132741228718*Lamb*r + 1759.2918860103

    a12=-12.566370614359*r**2 

    a21=-12.566370614359*r**2 

    a22=0

    return jnp.array(([a11,a12],[a21,a22]),dtype=float)

def Lagrangiano(r,Lamb):
    return  -Lamb*(-25 + 4.1887902047864*r**3) + 1300 + 879.64594300514*r**2


# Parametros
m = 50
tol = 1e-3
P0 = jnp.array([radioini,lmini], dtype=float)
################RUTINA009##########################
for i in range(m):
    F = sistema(P0)
    J = Jacob(P0)
    deltaP = jnp.linalg.solve(J, -F)
    P = P0 + deltaP
    err = jnp.linalg.norm(deltaP)
    relerr = err / jnp.linalg.norm(P)
    f_norm = jnp.linalg.norm(F)
    if (err < tol) or (relerr < tol) or (f_norm < tol):
        P0 = P
        break
    P0 = P

print("Analsis de convergencia:")
print("Error absoluto: ", err)
print("Error relativo: ", relerr)
print("Cantidad de iteraciones: ", i+1)
print("Valor de |F(P)|: ", f_norm)

dim_r = P[0]
L_res = jnp.abs(P[1])
C_res = Lagrangiano(dim_r,0)

print("------------------")
print("a) Radio de la cosa: ", jnp.around(dim_r,6))
print("c) Valor del multiplicador de Lagrange: ", jnp.around(L_res,6))
print("d) Costo del la cosa: ", jnp.around(C_res,6))