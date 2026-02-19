#Ax=B con Siedel
import numpy as np
np.set_printoptions(precision=6, suppress = True)


A = np.array([
[35,1,7,1,7,1],
[5,42,6,5,6,5],
[3,4,30,3,4,3],
[1,2,1,10,2,1],
[4,2,4,2,35,4],
[5,3,5,3,5,25]], dtype = float)

B = np.array([4000,3000,2000,1000,6000,5000],dtype= float)

X = np.array([1,2,3,4,5,9],dtype=float)
m = 50
tol = 0.1
n = len(B)
P = X.copy()

###########Rutina 003########### ##GAUS SIEDEL
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



print("\n Nro de iteraciones: ", i+1)
print("\n X: ",X)
print(f" Error:  {err: .6f}")
print(f"Error relativo: {relerr: .6f}")
