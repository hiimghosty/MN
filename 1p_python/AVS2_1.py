# Ejercicio de la concentracion del medicamento, por newton rhapson
import jax.numpy as jnp
from jax import grad

e = jnp.e
# c(t) = A*t*e^(-t/3), pero en realidad nos da de dato concentracion MAXIMA, por ende hallamos t para
# c(t)' = 0
concentracionMaxima = 1.4


def f(x):
    return 1 - x / 3


df = grad(f)
tol = 1e-3
p0 = 1.0
n = 100

for i in range(n):
    p = p0 - f(p0) / df(p0)
    err = jnp.abs(p - p0)
    relerr = jnp.abs(err / p)
    if tol > err or tol > relerr or tol > jnp.abs(f(p)):
        break
    p0 = p


print("\n El tiempo en que se llega  la concentracion maxima es: ", p)
A = (concentracionMaxima) / (p * jnp.exp(-p / 3))
print("\n La A es: ", A)

# PARTE 2

# c. Calculo de la cantidad adicional (en mg)
# Si A da una concentracion maxima de 1.4, necesitamos saber cuanto inyectar para subir
# lo que falta desde 0.29 hasta 1.4
concentracion_caida = 0.29
concentracion_faltante = concentracionMaxima - concentracion_caida
A_extra = A * (concentracion_faltante / concentracionMaxima)

print(f"\n c. La cantidad adicional de medicamento a inyectar es: {A_extra:0.6f}  ")


# d. Instante en el que se debería aplicar la segunda inyección (en horas)
# Buscamos cuando la concentracion cae a 0.29, usando p0 = 10
def f2(x):
    return A * x * jnp.exp(-x / 3) - concentracion_caida


df2 = grad(f2)
p0_2 = 10.0  # El valor inicial dado para esta parte

for i in range(n):
    p2 = p0_2 - f2(p0_2) / df2(p0_2)
    err2 = jnp.abs(p2 - p0_2)
    relerr2 = jnp.abs(err2 / p2)
    if tol > err2 or tol > relerr2 or tol > jnp.abs(f2(p2)):
        break
    p0_2 = p2

print(
    f"\n d. El instante en que la concentracion cae a 0.29 es a las (horas): {p2:0.6f}"
)
