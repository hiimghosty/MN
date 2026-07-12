import numpy as np
import jax.numpy as jnp
from jax import grad, config, jacfwd
config.update("jax_enable_x64", False)


VolumenMaximo = 1000 #m3
CostoCilindro = 10
CostoSemiesfera = 25
CostoBase = 30

def AreaCilindro(radio, altura):
    return  2 * jnp.pi * radio * altura

def AreaBase(radio):
    return jnp.pi * radio**2

def AreaSemiesfera(radio):
    return 2 * jnp.pi * radio**2    

def VolumenTotal(radio, altura):
    return AreaBase(radio) * altura + ( 4/3 * jnp.pi * radio**3) / 2

def lagrange(var):
    radio, altura, landa = var
    return (CostoCilindro * AreaCilindro(radio, altura) + CostoSemiesfera * AreaSemiesfera(radio) + CostoBase * AreaBase(radio) - landa * (VolumenTotal(radio, altura) - VolumenMaximo))

def costo(radio,altura):
    return (CostoCilindro * AreaCilindro(radio, altura) + CostoSemiesfera * AreaSemiesfera(radio) + CostoBase * AreaBase(radio))

m=20
tol=1e-1

sistema=grad(lagrange)
jacob_sist=jacfwd(sistema)


P0 = jnp.array([4, 15, -5],dtype=float) #radio, altura, landa



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
radioOptimo=P[0]
alturaOptima=P[1]
print(f"Costo optimo: {costo(radioOptimo,alturaOptima):.6f}")
