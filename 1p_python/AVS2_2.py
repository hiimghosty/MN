import numpy as np


def f(x):
    return 18.67 * np.exp(x - 6.2) - 802


def g(x):
    return x**3.32 + 4.5 * x - 802


def h(x):
    return 13.51 * (x**2.1) - 802


# Evaluamos quién llega (Bolzano)
print("Comprobación de signos (A, B, C):", f(9.2) * f(0), g(9.2) * g(0), h(9.2) * h(0))

tol = 1e-2
n = 100

# === RUTINA MOVIL B ===
a = 0.0
b = 9.2
c0 = a
for i in range(n):
    c = (b * g(a) - a * g(b)) / (g(a) - g(b))
    err = np.abs(c0 - c)
    relerr = np.abs(err / c)
    if tol > err or tol > relerr or tol > np.abs(g(c)):
        break
    else:
        if g(a) * g(c) < 0:
            b = c
        else:
            a = c
    c0 = c

tiempoB = c

# === RUTINA MOVIL C ===
# IMPORTANTE: Reiniciar variables
a = 0.0
b = 9.2
c0 = a
for i in range(n):
    c = (b * h(a) - a * h(b)) / (h(a) - h(b))
    err = np.abs(c0 - c)
    relerr = np.abs(err / c)
    if tol > err or tol > relerr or tol > np.abs(h(c)):
        break
    else:
        if h(a) * h(c) < 0:
            b = c
        else:
            a = c
    c0 = c

tiempoC = c

# RESPUESTAS 1, 2 y 3
print("\n--- RESULTADOS ---")
print(f"Tiempo Móvil B: {tiempoB:.6f} min")
print(f"Tiempo Móvil C: {tiempoC:.6f} min")

if tiempoC < tiempoB:
    print("Ítem 1: El primer móvil en terminar fue el C (cargar 3)")
    print(f"Ítem 2: Ocurrió en el tiempo: {tiempoC:.6f} minutos")
else:
    print("Ítem 1: El primer móvil en terminar fue el B (cargar 2)")
    print(f"Ítem 2: Ocurrió en el tiempo: {tiempoB:.6f} minutos")

difTiempos = np.abs(tiempoB - tiempoC)
print(f"Ítem 3: Diferencia de tiempos entre 1ro y 2do: {difTiempos:.6f} minutos")


# RESPUESTA 4: Velocidad del segundo (Asumiendo que es B)
def velocidadB(x):
    return 3.32 * (x**2.32) + 4.5


print(f"Ítem 4: Velocidad del 2do lugar 4 min antes: {velocidadB(tiempoB - 4):.6f}")

# RESPUESTA 5: Distancia del más atrasado (Móvil A) a los 9.2 minutos
# Distancia a la meta = Meta(802) - DistanciaRecorrida(9.2)
distancia_recorrida_A = 18.67 * np.exp(9.2 - 6.2)
distancia_faltante_A = 802 - distancia_recorrida_A
print(
    f"Ítem 5: Distancia a la meta del más atrasado (A): {distancia_faltante_A:.6f} hm"
)

# RESPUESTA 6: Tiempo extra para A
a = 0.0
b = 9.2
# Buscamos un b donde f(b) sea positivo para asegurar cambio de signo
while f(a) * f(b) > 0:
    b += 0.1

# Ahora sí aplicamos Falsa Posición para el Móvil A
c0 = a
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

tiempoA = c
tiempo_extra = tiempoA - 9.2
print(
    f"Ítem 6: El tiempo límite debe extenderse mínimamente {tiempo_extra:.6f} minutos"
)
