##SEMANA 2 AULA VIRTUAL
import numpy as np
np.set_printoptions(precision=6, suppress=True) #Pide 6 decimales y sin formato exponencial

A = np.array([
    [25, 12,  7],   # Metal:   
    [ 4, 12,  3],   # Plastico: 
    [15, 12, 34]    # Caucho:  
], dtype=float)

B = np.array([9440, 5710, 14890], dtype=float)

tol = 1e-2
m = 50
n = len(B)
X= np.zeros(n)
P = X.copy()
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

## para ver cantidades de metal

# como reordenamos para aplicar jacobi, volvemos al orden original
x1, x2, x3 = X[2], X[1], X[0]
x = np.array([x1, x2, x3], dtype=float)

# pide tomar parte entera, tiene sentido tomar piso para no sobre-estimar
x_int = np.floor(x).astype(int)

# dato del problema, consumo de metal por componente [7, 12, 25]
A_metal = np.array([7, 12, 25], dtype=float)
metal_disp_g = 9440.0

metal_usado_g = A_metal @ x_int
saldo_metal_g = metal_disp_g - metal_usado_g
saldo_metal_kg = saldo_metal_g / 1000.0

print("\nProducción real (enteros) [x1, x2, x3] =", x_int)
print(f"Metal usado: {metal_usado_g:.0f} g")
print(f"Saldo de METAL: {saldo_metal_kg:.6f} kg  (positivo=sobra, negativo=falta)")


print(A[0][:] @ x_int)