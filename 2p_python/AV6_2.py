# EJERCICIO SEMANAL
import numpy as np

# Debo hallar k para que el area de
# 5*x^2+(2*k+10)*x+2*k+31 sea igual a 100

coefConocidos = np.array([5, 10, 31], dtype=float)
coefK = np.array([0, 2, 2], dtype=float)

# Armo los polinomios
PoliniomioConocido = np.poly1d(coefConocidos)
PoliniomioDeK = np.poly1d(coefK)

f = PoliniomioConocido
q = PoliniomioDeK


def determinarK(a, b, n):
    ################RUTINA015########################## # SIMPSON 1/3
    A = 0.0
    B = 0.0
    h = (b - a) / n
    for xi in np.arange(a, b, 2 * h):
        A = A + (h / 3) * (f(xi) + 4 * f(xi + h) + f(xi + 2 * h))
        B = B + (h / 3) * (q(xi) + 4 * q(xi + h) + q(xi + 2 * h))

    k = (Areadada - A) / B
    return k, B


def areaPorSimpson(a, b, n):
    ################RUTINA015########################## # SIMPSON 1/3
    A = 0.0
    h = (b - a) / n
    for xi in np.arange(a, b, 2 * h):
        A = A + (h / 3) * (
            polinomioVerdadero(xi)
            + 4 * polinomioVerdadero(xi + h)
            + polinomioVerdadero(xi + 2 * h)
        )

    return A


# Areadada = Areaconocida + Areadek*K

# K = (Areadada - Areaconocida   )/Areadek

Areadada = 100
tol = 1e-2
puntoA = 7.0
puntoB = 9.0
n = 10
# Determinar K
k, B = determinarK(puntoA, puntoB, n)

print(f"El valor de k es: {k:0.6f}")
# Polinomio verdadero
# 5*x^2+(2*k+10)*x+2*k+31 sea igual a 100
coefVerdaderos = np.array([5, (2 * k + 10), (2 * k + 31)], dtype=float)
polinomioVerdadero = np.poly1d(coefVerdaderos)
Areaverdadera = areaPorSimpson(puntoA, puntoB, n)
print(f"El area para esa k es: {Areaverdadera:0.6f}")
# ERROR REAL EN EL AREA
errorArea = abs(Areadada - Areaverdadera)
print(f"El error real en el area es: {errorArea:0.6f}")

# ERROR MAXIMO EN EL AREA DEBIDO A LA TOLERANCIA
errorMaxArea = abs(B) * tol
print(f"El error maximo en el area por la tolerancia es: {errorMaxArea:0.6f}")
