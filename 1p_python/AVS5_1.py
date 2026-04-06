import numpy as np

# Tiempo ($seg$) | 2.4 | 3 | 3.4 | 4.2 | 4.7 | 5 | 5.5
# :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---:
# Producción ($kg$) | 6 | 6.23 | 7 | 7.13 | 7.74 | 8 | 7.2
x = np.array(([2.4, 3, 3.4, 4.2, 4.7, 5, 5.5]), dtype=float)
y = np.array(([6, 6.23, 7, 7.13, 7.74, 8, 7.2]), dtype=float)

n = len(x)
p = np.poly1d([1, -x[0]])
P = np.poly1d([y[0]])
for i in range(1, n):
    a = (y[i] - P(x[i])) / p(x[i])
    P = np.polyadd(P, a * p)
    p = np.polymul(p, np.poly1d([1, -x[i]]))

print("Polinomio interpolador:")
print(P)  ## Muy reundante pero este ya es el polinomio, NO LOS COEFICIENTES!!!!!!!!!

# Calcular polinomio derivado para el punto máximo
dP = P.deriv()


# Definimos f(x) = 0
def f(x):
    return dP(x)


# Definir x = g(x) para el metodo del punto fijo
# g(x) = -(dP(x) - dP[1]*x)/dP[1]
def g(x):
    return -(f(x) - dP[1] * x) / dP[1]


# Parametros de inicializacion
a = 4.7
b = 5.5
c0 = a
tol = 1e-8
n = 100
################RUTINA005##########################
for i in range(n):
    c = (b * f(a) - a * f(b)) / (f(a) - f(b))
    err = np.abs(c0 - c)
    relerr = np.abs(err / c)
    if tol > err or tol > relerr or tol > np.abs(f(c)):
        break
    else:
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    c0 = c

print("---Analisis de convergencia---")
print("Error absoluto:", err)
print("Error relativo:", relerr)
print("Error absoluto:", np.abs(f(c)))
print("Numero de iteraciones:", i + 1)

print("---Resultados---")
print("a) La maxima produccion es: ", P(c))
print("b) El tiempo maxima produccion es: ", c)
print("d) La produccion a los 2.93 seg. es: ", P(2.93))


# Para la produccion de 6.53kg
# Definimos f(x) = 0
def f(x):
    return P(x) - 6.53


# Definimos x = g(x)
def g(x):
    return -(f(x) - dP[1] * x) / dP[1]


# Parametros de inicializacion
a = 3
b = 3.4
c0 = a
tol = 1e-8
n = 10
################RUTINA005##########################
for i in range(n):
    c = (b * f(a) - a * f(b)) / (f(a) - f(b))
    err = np.abs(c0 - c)
    relerr = np.abs(err / c)
    if tol > err or tol > relerr or tol > np.abs(f(c)):
        break
    else:
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    c0 = c

print("---Analisis de convergencia---")
print("Error absoluto:", err)
print("Error relativo:", relerr)
print("Error absoluto:", np.abs(f(c)))
print("Numero de iteraciones:", i + 1)

print("---Resultados---")
print("c) El tiempo para producir 6.53kg es:", c)
