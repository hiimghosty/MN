import numpy as np

# Constantes
Q = 20
g = 9.81

# Definicion de la funcion f(x) = 0
def B(y):
    return 3+y

def Ac(y):
    return 3*y+(y**2)/2

def f(x):
    return 1-((Q**2)*B(x))/(g*(Ac(x)**3))

# Parametros de inicializacion
tol = 0.01
n = 10
a = 0.5
b = 2.5
c0 = a

# %%  RUTINA004##########################
for i in range(n):
 c=(a+b)/2
 err=np.abs(c0-c)
 relerr=err/np.abs(c)
 if tol>err or tol>relerr or tol>np.abs(f(c)):
   break
 else:
   if f(a)*f(c)<0:
     b=c
   else:
     a=c
 c0=c

print("Verificacion de convergencia: ")
print("error absoluto: ",err)
print("error relativo: ",relerr)
print("valor de f(c): ",np.abs(f(c)))
print("numero de iteraciones: ",i+1)

print("--------------------")
print("a) Cantidad de iteraciones: ",i+1)
print("b) El valor de la y: ", np.around(c,6))
print("c) El valor de B: ",np.around(B(c),6))
print("d) El valor de Ac: ",np.around(Ac(c),6))