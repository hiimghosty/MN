import numpy as np
import os

# Presentación (no afecta los cálculos)
np.set_printoptions(precision=6, suppress=True)
os.system("cls" if os.name == "nt" else "clear")

# Sistema Ax = b
A = np.array([[4, -1,  1],
              [4, -8,  1],
              [-2,  1,  5]], dtype=float)

b = np.array([7, -21, 15], dtype=float)

# Parámetros
tol = 0.01
max_iter = 50

# Chequeos mínimos
n = A.shape[0]
if A.shape[1] != n:
    raise ValueError("A debe ser cuadrada (n x n).")
if b.shape[0] != n:
    raise ValueError("b debe tener dimensión n.")
if np.any(np.diag(A) == 0):
    raise ValueError("Hay ceros en la diagonal de A: Jacobi no puede dividir por a_jj.")

# Inicialización: x^(0)
p = np.zeros(n, dtype=float)  # vector viejo (x^k)
x = np.zeros(n, dtype=float)  # vector nuevo (x^(k+1))

# Jacobi
for k in range(max_iter):
    for j in range(n):
        # suma_{i != j} a_{j,i} * p_i
        suma = 0.0
        for i in range(n):
            if i != j:
                suma += A[j, i] * p[i]

        # x_j^(k+1) = (b_j - suma) / a_jj
        x[j] = (b[j] - suma) / A[j, j]

    # Error entre iteraciones
    err = np.linalg.norm(x - p)
    normx = np.linalg.norm(x)
    relerr = err / normx if normx != 0 else np.inf

    # Criterio de parada
    if err < tol or relerr < tol:
        break

    # Actualizar: x^k <- x^(k+1)
    p = x.copy()

print("solución del sistema:\n", x)
print("cantidad de iteraciones realizadas:", k + 1)
print("err:", err, "relerr:", relerr)
