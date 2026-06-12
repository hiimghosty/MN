# TAA DE FILAMENTOS

import numpy as np

Temperaturas = np.array([105, 108, 157, 168, 183, 218, 221, 236], dtype=float)
kp = np.array([37.24, 51.19, 66.07, 79.33, 34.28, 60.96, 87.91, 13.31], dtype=float)
ki = np.array([2.356, 1.667, 1.25, 1.07, 3.04, 3.981, 0.763, 3.129], dtype=float)
kd = np.array([51.96, 96.57, 85.93, 64.76, 81.82, 68.91, 86.3, 60.96], dtype=float)


temperaturaPLA = 170
temperaturaPTG = 180
temperaturaABS = 190
tol = 1e-8
# Me pide hallar los nuevos parametros PID si se desea imprimir en PLA
# Debo hallar entonces 3 polinomios interpoladores, la temperatura de pla
# Que es 170 no esta directamente tabulada


def interpolacionLagrange(x, y):
    ################RUTINA012-Interpolador de Lagrange##########################
    #
    n = len(x)
    P = np.poly1d([0])
    # rutina:
    for i in range(n):
        a = np.delete(np.arange(n), i)
        p = np.poly1d([1, -x[a[0]]])
        for j in range(1, n - 1):
            p = np.polymul(p, np.poly1d([1, -x[a[j]]]))
        P += y[i] * p / p(x[i])
    return P


polinomioKp = interpolacionLagrange(Temperaturas, kp)
polinomioKd = interpolacionLagrange(Temperaturas, kd)
polinomioKi = interpolacionLagrange(Temperaturas, ki)

# Estos polinoimios me dan esos parametros para una temperatura en celsius
# Ahora debo evaluarlos en temperaturaPLA para saber el parametro

print(f"Nuevo valor Kp para PLA {polinomioKp(temperaturaPLA):.6f}")
print(f"Nuevo valor Ki para PLA {polinomioKi(temperaturaPLA):.6f}")
print(f"Nuevo valor Kd para PLA {polinomioKd(temperaturaPLA):.6f}")

# Ahora pide hallar el valor maximo de ki, el valor maximo de ki lo obtenemos
# Igualando su derivada a 0 y hallando ese valor

################RUTINA005-Falsa Posicion##########################
#
f = polinomioKi.deriv()

a = 200
b = 210
c0 = a
n = 100
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
temperaturadelmaximo = c
maximoKi = polinomioKi(temperaturadelmaximo)

print(f"Valor maximo de ki es {maximoKi:.6f}")
print(f"Temperatura a la que se da este maximo {temperaturadelmaximo:.6f}")
