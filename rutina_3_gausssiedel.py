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