import numpy as np

coefDelPolinomioQueNoDependeDeK=np.array([5,10,31],dtype=float)
coefDelPolinomioQueSiDependeDeK=np.array([0,2,2],dtype=float)





PolinomioQueNoDependeDeK = np.poly1d(coefDelPolinomioQueNoDependeDeK)
PolinomioQueSiDependeDeK = np.poly1d(coefDelPolinomioQueSiDependeDeK)

AreaTotal=100

f = PolinomioQueNoDependeDeK
q = PolinomioQueSiDependeDeK
a = 7
b = 9
n = 10



################RUTINA015########################## ## REGLA DE SIMPSON 1/3
h=(b-a)/n
AreaDelPolinomioQueNoDependeDeK=0
AreaDelPolinomioQueSiDependeDeK=0
y=f(np.linspace(a,b,n+1))
z=q(np.linspace(a,b,n+1))
k=0
while(k<n):
    AreaDelPolinomioQueNoDependeDeK+=(h/3)*(y[k]+4*y[k+1]+y[k+2])
    AreaDelPolinomioQueSiDependeDeK+=(h/3)*(z[k]+4*z[k+1]+z[k+2])

    KPedida= (AreaTotal-AreaDelPolinomioQueNoDependeDeK)/AreaDelPolinomioQueSiDependeDeK
    k+=2



print(f"a) El valor de K es: {KPedida:.6f}")
polinomioVerdadero=PolinomioQueNoDependeDeK+KPedida*PolinomioQueSiDependeDeK
print(f"b) El polinomio verdadero es: {polinomioVerdadero}")

a = 7
b = 9
n = 10
f = polinomioVerdadero
################RUTINA015########################## ## REGLA DE SIMPSON 1/3
h=(b-a)/n
A=0
y=f(np.linspace(a,b,n+1))
k=0
while(k<n):
    A+=(h/3)*(y[k]+4*y[k+1]+y[k+2])
    k+=2
###########
print(f"c) El área bajo la curva es: {A:.6f}")
errorreal=abs(A-AreaTotal)
print(f"d) El error real es: {errorreal:.6f}")



## OTRO METODO


def polinomioQueNoDependeDeK(x):
    return 5 * (x**2) + 10*x + 31

def polinomioQueSiDependeDeK(x):
    return 2 * x + 2    
