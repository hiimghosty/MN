import numpy as np

x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
densidad = np.array([2.3, 2.43, 2.6, 3, 3.4, 3.5, 3.8, 4.2, 4.38, 5.2, 5.7])
Area = np.array([2, 3.24, 2.3, 2.2, 1.68, 1.3, 1, 0.86, 0.75, 0.7, 0.66])

def simpson_13(x, y):
    n = len(x) - 1

    if n % 2 != 0:
        raise ValueError("Para Simpson 1/3, n debe ser par.")

    a = x[0]
    b = x[-1]
    h = (b - a) / n

    A = 0
    k = 0

    while k < n:
        A += (h / 3) * (y[k] + 4*y[k+1] + y[k+2])
        k += 2

    return A

# Volumen
Volumen = simpson_13(x, Area)
print(f"El volumen es: {Volumen:0.6f}")

# Masa
Masa = simpson_13(x, Area * densidad)
print(f"La masa es: {Masa:0.6f}")

# Abscisa del centro geométrico (centroide, densidad uniforme)
Centro_geometrico = simpson_13(x, x * Area) / Volumen
print(f"La abscisa del centro geométrico es: {Centro_geometrico:0.6f}")

# Abscisa del centro de masa (pondera por la densidad real)
Centro_masa = simpson_13(x, x * Area * densidad) / Masa
print(f"La abscisa del centro de masa es: {Centro_masa:0.6f}")

