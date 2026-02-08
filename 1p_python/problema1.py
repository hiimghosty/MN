import numpy as np
##Decia algo de plasticos y metales, sistema 3x3
A = np.array([[7, 12,  26],
              [3, 12,  4],
              [35,  12,  15]], dtype=float)

B = np.array([9570, 5710, 15140], dtype=float) 

C = np.hstack([A, B.reshape(-1,1)])

delta = 1e-12 

###########Rutina 001###########
if np.abs(np.linalg.det(A)) < delta: #Segun CHATGPT no es buen criterio para detectar matriz singular
        print('el sistema no tiene solucion o tiene infinitas soluciones')
else:
    n=A.shape[0]
    j = -1
    i = n - 1
    while j < n-2:
        if i == n - 1:
            j += 1
            i = j
            subcol = np.abs(C[j:, j])
            t = np.argmax(subcol)
            pibot = subcol[t]
            if delta > pibot:
                print('el sistema no tiene solucion o tiene infinitas soluciones')
            else:
                aux=C[j,:].copy()
                C[j,:]=C[t+j,:]
                C[t+j,:]=aux.copy()
        i += 1
        k = C[i, j] / C[j, j]
        C[i, j:] = C[i, j:] - k * C[j, j:]
X=np.zeros((n,1),dtype=float)
X[n-1] = C[-1, -1] / C[n-1, n-1]
for i in range(n-2, -1, -1):
        X[i] = (C[i, -1] - C[i, i+1:n].dot(X[i+1:n])) / C[i, i]

print("X =\n", X)          # <- acá está tu vector solución
print("X plano =", X.ravel())  # opcional: como vector 1D
print("Chequeo A@X =", (A @ X).ravel())  # debería parecerse a B