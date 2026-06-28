import numpy as np
import jax.numpy as jnp
from jax import grad, jacfwd


# DATOS DEL EJERCICIO 
g = 32.17

# FUNCION DEL EJERCICIO
def f(x):
  return (-g/(2*jnp.pow(x, 2)))*(((jnp.exp(1.2*x)-jnp.exp(-1.2*x))/2) - jnp.sin(1.2*x)) -1.9

# PARAMETROS DE INICIALIZACION
n = 30
p0 = 1.0
tol = 1e-3
df = grad(f)



################RUTINA006-Newton-Rapson##########################
for i in range(n):
 p=p0-f(p0)/df(p0)
 err=jnp.abs(p-p0)
 relerr=jnp.abs(err/p)
 if tol>err or tol>relerr or tol>jnp.abs(f(p)):
   break
 p0=p


# 1) EL VALOR DE LA VELOCIDAD ANGULAR
print("El valor de la velocidad angular es: ", np.around(p, 6))

# 2) LA CANTIDAD DE ITERACIONES
print("Cantidad de iteraciones: ", i+1)

# 3) ERROR ABSOLUTO
print("El error absoluto es: ", np.around(err, 6))

# 4) ERROR COMETIDO = ERROR RELATIVO
print("El error cometido es: ", np.around(relerr, 6))

# 5) PARA EL VALOR ENCONTRADO, HALLAR EL VALOR DE LA ORDENADA DE LA FUNCION f=x(t)+k*g
ordenada = f(p)*(np.pow(10, 6))
print("La ordenada de la funcion es: ", np.around(ordenada, 6))