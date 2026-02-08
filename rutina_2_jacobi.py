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