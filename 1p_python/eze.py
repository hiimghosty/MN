import numpy as np
# Ax=B
A = np.array([
    [18,  4,  4,  2,  4],   # 18>= (4+4+2+4), CUMPLE
    [ 4, 24,  4,  4,  2],   # 24 >+ (4+4+4+2), CUMPLE
    [ 2,  2, 30,  4,  4],   
    [ 4,  4,  2, 36,  1],  
    [ 2,  2,  2,  2, 42]    
], dtype=float)

B = np.array([150, 200, 100, 300, 250], dtype=float) #vector 1D

m = 50 #nro maximo de iteraciones
n = len(B) #length , longitud 
tol = 1e-4 #10 a la -4
X = np.zeros(n) #vector inicial, suele ser 0s
P = X.copy()



###########Rutina 002###########  ##JACOBI
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

print(i+1) #nro de iteraciones
print(relerr) #error relativo
print(err) #error absoluto

print(f"Error absoluto {err:.6f}") 
print(f"Error relativo {relerr:.6f}")
