# Ejercicio del plano inclinado y eso
import jax.numpy as jnp
from jax import grad

x_val = 1.7
t = 1
g = 32.17
# Tenemos de dato una x(t), pero en realidad queremos hallar la w para esto


def f(x):
    return (-g / (2 * (jnp.pow(x, 2)))) * (
        (jnp.exp(x * t) - jnp.exp(-x * t)) / 2 - jnp.sin(x * t)
    ) - x_val


df = grad(f)

tol = 1e-5
p0 = jnp.array(1.0)
n = 100
# Newton - rhapson


for i in range(n):
    p = p0 - f(p0) / df(p0)
    err = jnp.abs(p - p0)
    relerr = jnp.abs(err / p)
    if tol > err or tol > relerr or tol > jnp.abs(f(p)):
        break
    p0 = p


print("Analisis de Convergencia: ")
print("error absoluto: ", err)
print("error relativo: ", relerr)
print("el valor de f(c): ", jnp.abs(f(p)))
print("numero de iteraciones: ", i + 1)

print("----------------")
print("a) Cantidad de iteraciones: ", i + 1)
print("b) El valor de w es: ", jnp.round(p0, 6))
print("c) El error absoluto: ", jnp.round(err, 6))
print("d) El error relativo: ", jnp.round(relerr, 6))
