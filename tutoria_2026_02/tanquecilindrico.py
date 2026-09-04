import numpy as np
from jax import grad,jacfwd
import jax.numpy as jnp

espesor=0.35 # cm
volumenTotal=950 # cm3


def volumenDeBase(r,e):
    R=r+e
    return ((np.pi*(R**2)) * e) * 2 # tapa y base

def volumenParedes(r,h,e):
    R=r+e
    volumenCilindroGrande = (np.pi * (R**2) ) * h
    volumenCilindroChico= (np.pi * (r**2)) * h
    return volumenCilindroGrande - volumenCilindroChico


def lagrange(var):
    r, h, landa = var
    return volumenDeBase(r,espesor) + volumenParedes(r,h,espesor) - landa * (np.pi*(r**2)*h - volumenTotal)

m=20
tol=1e-4

sistema=grad(lagrange)
jacob_sist=jacfwd(sistema)


P0 = jnp.array([5.5, 10.5, 0.45],dtype=float) #radio, altura, landa



###################################################
################RUTINA009########################## #NEWTON RHAPSON MULTI DIMENSIONAL
for i in range(m):
    F=sistema(P0)
    J=jacob_sist(P0)
    deltaP=jnp.linalg.solve(J,-F)
    P=P0+deltaP
    err=jnp.linalg.norm(P-P0)
    relerr=err/jnp.linalg.norm(P)
    F_norm=jnp.linalg.norm(sistema(P))
    if tol>err or tol>relerr or tol>F_norm:
        break
    else:
        P0=P.copy()



print(f"Iteraciones: {i+1}")
print(P)
radioOptimo = P[0]
alturaOptima = P[1]

print("Radio óptimo:", radioOptimo)
print("Altura óptima:", alturaOptima)