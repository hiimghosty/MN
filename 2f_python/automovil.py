import numpy as np
k=0
tiempo=np.array([0,3.5,7,10.5,14,21,28,31.5,35,45.5,56,63,70,73.5,77],dtype=float)
velocidad=np.array([123,116,104,88,77,86,0,108,121,134,147,0,148,134,124],dtype=float)
velocidadk=np.array([0,0,0,0,0,0,2,0,0,0,0,3,0,0,0],dtype=float)
print(len(tiempo),len(velocidad),len(velocidadk))

longitudTotalPista=8000

def simpson_13(a,b,f):
    n=len(f)-1
    #h=(b-a)/n
    A=0
    y=f 
    k=0
    while k<n:
        h=tiempo[k+1]-tiempo[k]
        A+=(h/3)*(y[k]+4*y[k+1]+y[k+2])
        k+=2
    return A

a=tiempo[0]
b=tiempo[-1]
valordeK=(longitudTotalPista-simpson_13(a,b,velocidad))/simpson_13(a,b,velocidadk)
print(valordeK)
velocidadVerdadera=velocidad+valordeK*velocidadk
# ahora debemos hallar velocidad media para t=35
a=tiempo[0]
b=tiempo[8]
recorridohasta35=simpson_13(a,b,velocidadVerdadera)
velocidadmedia=recorridohasta35/(b-a)
print("La velocidad media hasta t=35 es: ",velocidadmedia)  