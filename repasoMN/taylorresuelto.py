###################################################
################LIBRERIAS##########################
import numpy as np
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", False)
from jax import jacfwd, grad


# Datos
x_inicial=1.0
y_inicial=3.0
x0=x_inicial
y0=y_inicial
h=0.1


################RUTINA021##########################
########MANUAL##########
def dy(x, y):
    return 6/y
def d2y(x, y):
    # (dy)'_x + (dy)'_y * dy
    return grad(dy, argnums=0)(x, y) + grad(dy, argnums=1)(x, y) * dy(x, y)
# def d3y(x, y):
#     # (dy)'_x + (dy)'_y * dy
#     return grad(d2y, argnums=0)(x, y) + grad(d2y, argnums=1)(x, y) * dy(x, y)
# def d4y(x, y):
#     # (dy)'_x + (dy)'_y * dy
#     return grad(d3y, argnums=0)(x, y) + grad(d3y, argnums=1)(x, y) * dy(x, y)
# def d5y(x, y):
#     # (dy)'_x + (dy)'_y * dy
#     return grad(d4y, argnums=0)(x, y) + grad(d4y, argnums=1)(x, y) * dy(x, y)

#Matriz de resultados
#Columnas: fila-tiempo-posicion-velocidad-aceleracion

Resultados=[]
velocidad=dy(x0, y0)
aceleracion=d2y(x0, y0)
Resultados.append([0,x0, y0, velocidad, aceleracion])
n=3
m=1
resultados = np.zeros((n + 1, 2 + m))

for i in range(0, 3):
    f1  = dy(x0, y0)
    f2  = d2y(x0, y0)
    # f3  = d3y(x0, y0)
    # f4  = d4y(x0, y0)
    # f5  = d5y(x0, y0)
    y1  = y0 + f1*h + f2/2 * h**2#+(f3*h**3)/6+(f4*h**4)/24+(f5*h**5)/120
    x1  = x0 + h
    y0  = y1
    x0  = x1

    #Matriz de resultados
    #Columnas: fila-tiempo-posicion-velocidad-aceleracion

    velocidad=dy(x0, y0)
    aceleracion=d2y(x0, y0)
    Resultados.append([i+1,x0, y0, velocidad, aceleracion])

print(Resultados    )