import numpy as np

print("gola")
A = np.array(
    [[60, 8, 19, 11], [14, 65, 15, 12], [11, 17, 60, 12], [15, 10, 6, 65]], dtype=float
)
A = A / 100
B = np.array([34, 52, 56, 59], dtype=float)
tol = 1e-2
m = 50
n = len(B)
X = np.zeros(n)
P = X.copy()
################RUTINA002-Gauss Jacobi###########################
for i in range(m):
    for j in range(n):
        X[j] = (
            B[j] - A[j, np.delete(np.arange(n), j)].dot(P[np.delete(np.arange(n), j)])
        ) / A[j, j]
    err = np.linalg.norm(X - P)
    normX = np.linalg.norm(X)
    relerr = err / normX
    if tol > err or tol > relerr:
        break
    else:
        P = X.copy()

print(X)
print(f"Proveedor A: {X[0]:.6f} kg")
print(f"Proveedor B: {X[1]:.6f} kg")
print(f"Proveedor C: {X[2]:.6f} kg")
print(f"Proveedor D: {X[3]:.6f} kg")
print(f"Error absoluto {err:.6f}")
