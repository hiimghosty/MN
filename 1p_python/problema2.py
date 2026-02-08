import numpy as np
#resuelto de carlosbeni
A = np.array([
            [60,8,19,11],
            [14,65,15,12],
            [11,17,60,12],
            [15,10,6,65]], dtype = float)

A = A/100
B= np.array([34,52,56,59],dtype=float)
n=len (B)
X = np.zeros(n)
P =X.copy()
m = 10
tol = 1e-2
###NECESARIAMENTE PARA GAUSS SIEDEL NECESITAMOS UNA MATRIZ DIAGONAL
###########Rutina 003###########
for i in range(m):
    for j in range(n):
        X[j] = (B[j] - A[j, np.delete(np.arange(n), j) ].dot(X[np.delete(np.arange(n), j) ])) / A[j, j]
    err = np.linalg.norm(X - P)
    normX = np.linalg.norm(X)
    relerr = err / normX
    if tol > err or tol > relerr:
        break
    else:
        P = X.copy()


print("X =\n", X)          # <- acá está tu vector solución
print("X plano =", X.ravel())  # opcional: como vector 1D
print("Chequeo A@X =", (A @ X).ravel())  # debería parecerse a B
print("\n", err)
print("\n",relerr)