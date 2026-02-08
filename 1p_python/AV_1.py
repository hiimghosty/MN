##SEMANA 1 AULA VIRTUAL
import numpy as np
np.set_printoptions(precision=6, suppress=True) #Pide 6 decimales y sin formato exponencial
A = np.array([
    [18,  4,  4,  2,  4],   
    [ 4, 24,  4,  4,  2],   
    [ 2,  2, 30,  4,  4],   
    [ 4,  4,  2, 36,  1],  
    [ 2,  2,  2,  2, 42]    
], dtype=float)

B = np.array([150, 200, 100, 300, 250], dtype=float)
tol = 1e-4
n = len(B)
m = 50 #Criterio real de parada en jacobi es el error no el nro de iteraciones
X = np.zeros(n) #me da un vector nulo como valor inicial
P = X.copy()
#PIDE POR JACOBI
###########Rutina 002########### 
for i in range(m):
    for j in range(n):
        X[j] = (B[j] - A[j, np.delete(np.arange(n), j) ].dot(P[np.delete(np.arange(n), j) ])) / A[j, j] #segun chatgpt el np.delete es caro
    err = np.linalg.norm(X - P)
    normX = np.linalg.norm(X)
    relerr = err / normX
    if tol > err or tol > relerr:
        break
    else:
        P = X.copy()

print(X)
print(f"Error absoluto {err:.6f}") #la opcion setprintoptions solo controla elementos de numpy, por ended este formateo es el mejor para forzar 6 decimales de precision
print(f"Error relativo {relerr:.6f}")
