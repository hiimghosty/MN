import numpy as np

np.set_printoptions(precision=6, suppress=True) #Pide 6 decimales y sin formato exponencial

#Intentar siempre meter la matriz como diagonal dominante

A = np.array([[34, 12, 15], [3, 12, 4], [7, 12, 25]], dtype=float)

B = np.array([14890, 5710, 9440])

tol = 1e-2

n = len(B) #n es la dimension del vector B

m = 50 #m es el numero maximo de interacciones 

X = np.zeros(n) #x es un vector de ceros de la misma dimension de B

P = X.copy() #vector auxiliar donde se guardaran los datos para la sgte interaccion


################RUTINA002##########################
for i in range(m):
    for j in range(n):          
        X[j] = (B[j] - A[j, np.delete(np.arange(n), j) ].dot(P[np.delete(np.arange(n), j) ])) / A[j, j]
    err = np.linalg.norm(X - P)
    normX = np.linalg.norm(X)
    relerr = err / normX
    if tol > err or tol > relerr:
        break
    else:
        P = X.copy()

print(X)

print(f"Error absoluto {err:.6f}") #la opcion setprintoptions solo controla elementos de numpy, por ended este formateo es el mejor para forzar 6 decimales de precision
print(f"Error relativo {relerr:.6f}") #sirve para imprimir directamente con 6 decimal, 6f es para tener solo con 6 decimales
metal_disponible = 9440
X_entera = np.floor(X).astype(int)
ConsumoMetales = A[2][:]
SobranteMetales =  metal_disponible - (ConsumoMetales  @ X_entera)
print(SobranteMetales)