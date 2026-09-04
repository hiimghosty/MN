import numpy as np

def deformacionBarra1(P):
    return -5.4*(P**(-2)) + P/90

def deformacionBarra2(P):
    return ((27*P**(4.3)) + 5.2)/(5.4*P + (P**5))

tol = 1e-4
distancia=7.2

def f(P):
    return deformacionBarra1(P) + deformacionBarra2(P) - distancia

limiteInferior = 400
a = limiteInferior
limiteSuperior = 600
b = limiteSuperior 
c0 = a
# ESTOS LIMITES NO CUMPLEN EL TEOREMA DE BOLZANO

n= 100
###################################################
################RUTINA004########################## # BISECCION
for i in range(n):
 c=(a+b)/2
 err=np.abs(c0-c)
 relerr=np.abs(err/c)
 if tol>err or tol>relerr or tol>np.abs(f(c)):
   break
 else:
   if f(a)*f(c)<0:
     b=c
   else:
     a=c
 c0=c

 
print("Iteraciones:", i+1)
print("P:", c)
print("f(P):", f(c))