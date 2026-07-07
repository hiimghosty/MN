# Clase de repaso MN — BLOQUE 1
## "Los errores que nadie te avisa (y cómo cazarlos en el parcial)"

**Duración:** ~25–30 min · **Material de apoyo:** `demos_bloque1.py` (corré los bloques en vivo)

**Objetivo del bloque:** que salgan de acá sabiendo (1) leer un error de Python en 5 segundos y (2) detectar los errores que NO tiran error y que son los que realmente reprueban.

---

## 0 · GANCHO — El demo que abre la clase (3 min)

> Corré **DEMO 1** de `demos_bloque1.py` en vivo, en silencio, sin explicar nada todavía.

Mostrás en pantalla una eliminación de Gauss resolviendo:

```
2x + y = 3
 x + 3y = 5
```

y sale:

```
respuesta: (0.75, 1.5)
```

Preguntá al público: **"¿Está bien esta respuesta?"**

Dejá que digan que sí (se ve perfecta, corrió sin ningún error). Entonces revelás que la respuesta real es **(0.8, 1.4)**. El código no tenía ni un error rojo. Solo faltó un `.astype(float)`.

**Frase de cierre del gancho (la que se tienen que llevar):**

> *"En el parcial, el error que te reprueba no sale en rojo. Sale en negro. El rojo te grita y te dice dónde. El negro te miente y te devuelve un número lindo. Hoy, primer bloque: cómo hacer que los rojos sean triviales, y cómo cazar los negros antes de que te cuesten el examen."*

---

## 1 · MARCO MENTAL — Python tiene dos tipos de error (2 min)

Pizarrón, dos columnas:

| 🔴 **Los que GRITAN** (traceback) | ⚫ **Los que MIENTEN** (silenciosos) |
|---|---|
| Python frena y te dice tipo + línea | Corre entero y devuelve un número mal |
| Molestos pero **fáciles**: te llevan al lugar | **Peligrosos**: parecen bien, no dicen nada |
| Ej: `NameError`, `IndexError`, `SyntaxError` | Ej: dtype int, columna vs fila, `.copy()` |

La moraleja contraintuitiva: **el error en rojo es tu amigo.** El que hay que temer es el que no aparece.

---

## 2 · LOS ROJOS — leer un traceback en 5 segundos (5 min)

**Regla de oro #1: el traceback se lee de ABAJO hacia arriba.**
- La **última línea** = tipo de error + mensaje (qué pasó).
- La línea de **archivo + número** justo arriba = dónde pasó.
- Todo el medio (líneas de numpy/jax) al principio **ignoralo**.

**Los 5 rojos que SÍ te van a salir en MN** (mostrá el mensaje real de cada uno):

1. **`NameError: name 'c0' is not defined`** → el más común de todos. Pegaste la rutina pero te olvidaste la *cabecera* (declarar `c0`, `P`, `df`, `delta`...). Las rutinas son fragmentos: sin cabecera, no viven.
2. **`IndexError: index 6 is out of bounds for axis 0 with size 6`** → Simpson con `n` que no corresponde, u off-by-one. (DEMO 5)
3. **`ValueError: ... could not be broadcast together with shapes ...`** → mezclaste formas de vector/matriz. `.shape` lo resuelve.
4. **`numpy.linalg.LinAlgError: Singular matrix`** → el jacobiano o `A` se volvió singular (típico en Newton multidim con mal `p0`).
5. **`SyntaxError: ...`** → falta un `(`, `[` o `:`.

**Regla de oro #2 (el tip que nadie te dice): el `SyntaxError` MIENTE con el número de línea.**
Casi siempre el paréntesis que falta está en la línea **de arriba** de la que te marca. En rutinas como Jacobi, con líneas larguísimas llenas de paréntesis anidados:

```python
X[j] = (B[j] - A[j, np.delete(np.arange(n), j)].dot(P[np.delete(np.arange(n), j)]) / A[j, j]
#      ^--- abrís acá                                                             falta un ) antes del /
```

Truco práctico: contá paréntesis de a pares, o partí la línea en variables intermedias.

---

## 3 · LOS NEGROS — el corazón del bloque (10 min)

Estos son los "tips que nadie te dice". Cada uno es un demo de 30 segundos.

### 3.1 · La bomba de enteros → SIEMPRE `.astype(float)` (DEMO 1)
Si la matriz aumentada queda en `int`, cada resta de fila se **trunca** a entero y arrastra el error hasta el final. Da **(0.75, 1.5)** en vez de **(0.8, 1.4)**, sin quejarse.
**Señal de alarma:** `print(C.dtype)` dice `int64`. **Antídoto:** `C = np.hstack((A, B.reshape(-1,1))).astype(float)`.

### 3.2 · Columna vs fila → el clásico de Jacobi/Gauss-Seidel (DEMO 2)
Si `B` o `P` quedan como columna `(n,1)` y el resto es fila `(n,)`, la resta `X - P` se **broadcastea** a una matriz `(n,n)`. La norma de convergencia deja de medir lo que creés: pasa de **0.15** (real) a **3.43** (fantasma). No tira error → itera de más, de menos, o corta cuando no debe.
**Señal:** `print(X.shape, P.shape)`. **Antídoto:** mantené TODO en 1D `(n,)` en las iterativas.

> **Caso real — el ejercicio de Jacobi del aula virtual (Semana 1).**
> En ese ejercicio, `B = np.array([150, 200, 100, 300, 250], dtype=float)` está en fila `(5,)` y converge en 12 iteraciones. Si en cambio lo declarás como columna:
> ```python
> B = np.array([[150],[200],[100],[300],[250]], dtype=float)   # shape (5,1)
> ```
> la MISMA rutina se rompe con:
> ```
> ValueError: setting an array element with a sequence.
> ```
> **Enseñanza doble:** el "columna vs fila" a veces es **negro** (se mete en la norma, como el DEMO 2) y a veces es **rojo** (revienta en la asignación `X[j] = ...`, como acá). Depende de dónde pega la forma equivocada. La defensa es la misma en los dos: `print(B.shape)` y, si te llega como columna, `B = B.flatten()`.

### 3.3 · Aliasing → `.copy()` no es opcional (DEMO 3)
En numpy, `P = X` **no copia**: son el mismo objeto. Si después modificás `X`, `P` cambia sola. En un bucle iterativo eso hace que `err = ||X - P|| = 0` en la primera vuelta y la rutina **corta al toque** devolviendo un resultado prematuro.
**Antídoto:** `P = X.copy()`. Ojo también con el *swap* de filas en Gauss: `aux = C[j,:].copy()`.

### 3.4 · `**` no `^` · y `np.sin` es en RADIANES (DEMO 4)
- `2 ^ 3` da **1** (es XOR de enteros, no potencia) y corre sin quejarse. Con float, `2.0 ^ 3` sí tira `TypeError`.
- `np.sin(30)` calcula el seno de **30 radianes** = −0.988, no de 30°. Si el problema viene en grados: `np.sin(np.radians(30))` = 0.5.

### 3.5 · Dos trampas más finas (mencionar, sin demo largo)
- **`argmax` es relativo al sub-arreglo.** En el pivoteo de Gauss, `t = np.argmax(np.abs(C[j:, j]))` te da el índice **dentro de `C[j:]`**, no la fila real. Por eso hay que usar `C[t+j, :]`. Olvidar el `+j` = pivote equivocado, silencioso.
- **No mezcles `np` y `jnp`.** Interpolación (`poly1d`) → numpy puro. `jnp` solo donde derivás (Newton, mínimos, jacobianos). `poly1d` con un *tracer* de jax dentro de `grad`/`jit` explota.

---

## 4 · LOS 4 COMANDOS QUE SALVAN EXÁMENES (3 min)

Poné esto en una sola diapo, bien grande:

1. **`print(x.shape)`** → caza columna-vs-fila y broadcasting (el 50% de los negros).
2. **`print(x.dtype)`** → caza la bomba de enteros. Si dice `int`, `.astype(float)`.
3. **Probá con un caso que sepas la respuesta** ANTES de confiar: raíz de `x²−4` es 2; un sistema 2×2 resuelto a mano. Si la rutina falla el caso de juguete, va a fallar en el parcial.
4. **`print` dentro del bucle** (`err`, `i`): si converge en 1 vuelta o nunca, lo ves al instante.

> Frase: *"Ante 'corre pero da mal', no adivines: `.shape` y `.dtype` primero. Dos líneas te encuentran el 80% de los bugs."*

---

## 5 · INTERACTIVO — "Cazá el bug en 60 segundos" (5 min)

Mostrá cada snippet, arrancá cronómetro, que griten dónde está.

**Bug A** (aliasing en el swap de Gauss):
```python
aux = C[j, :]          # <-- falta .copy(): aux apunta a la MISMA fila
C[j, :] = C[t+j, :]
C[t+j, :] = aux        # el swap se corrompe
```

**Bug B** (Newton que no avanza):
```python
for i in range(n):
    p = p0 - f(p0)/df(p0)
    err = abs(p - p0)
    if tol > err: break
    # <-- falta  p0 = p  : itera SIEMPRE desde el mismo punto
```

**Bug C** (Simpson con `n` mal contado):
```python
n = len(x)             # <-- deberia ser len(x) - 1 (subintervalos, no puntos)
h = (b - a)/n
```

Cerrá pidiéndoles que digan **cuál era rojo y cuál negro**. (A y B son negros: corren y dan mal. C puede caer en `IndexError`.)

---

## 6 · TEORÍA MÍNIMA — Bolzano: cómo elegir bien `a` y `b` (6 min)

*(Para bisección y falsa posición. Es la teoría que hace que esos métodos tengan sentido.)*

**El teorema en una línea:** si `f` es **continua** en `[a, b]` y `f(a)` y `f(b)` tienen **signos opuestos** (`f(a)·f(b) < 0`), entonces existe al menos un `c` en `(a, b)` con `f(c) = 0`.

Traducción para el parcial: **antes de tirar bisección, verificá que tu intervalo encierra una raíz.** Si no hay cambio de signo, el método no tiene nada garantizado.

Los tres casos (pizarrón):
- `f(a)·f(b) < 0` → cambio de signo → **hay al menos una raíz**. ✅ Válido.
- `f(a)·f(b) > 0` → mismo signo → **sin garantía** (puede haber 0 raíces… o 2, o 4). ❌ No sirve para bisecar.
- `f(a)·f(b) = 0` → `a` o `b` **ya es** la raíz.

> Ojo con el detalle fino: "mismo signo" **no prueba** que no haya raíz; puede haber un número **par** de raíces escondidas. Bolzano solo te garantiza cuando hay cambio de signo.

**Implementación simple — chequeo de validez:**
```python
import numpy as np

def es_valido(f, a, b):
    fa, fb = f(a), f(b)
    if fa == 0: return f"a={a} ya es raiz"
    if fb == 0: return f"b={b} ya es raiz"
    if np.sign(fa) != np.sign(fb):
        return f"OK: cambio de signo -> hay raiz en ({a}, {b})"
    return "NO valido: f(a) y f(b) tienen el mismo signo"

f = lambda x: x**3 - x - 2
print(es_valido(f, 1, 2))   # OK: hay raiz
print(es_valido(f, 2, 3))   # NO valido
```

**Tip que nadie te dice:** usá `np.sign(fa) != np.sign(fb)` en vez de `fa*fb < 0`. El **producto** puede **desbordar** (dar `inf`) o hacerse `0` por redondeo cuando `f(a)` y `f(b)` son muy grandes o muy chicos, y ahí el chequeo miente. Comparar signos nunca se pasa de rango. *(Es, literalmente, un "negro" del bloque anterior aplicado a la teoría.)*

### Cómo ubicarlo en un gráfico
La raíz es donde la curva **cruza el eje x** (donde `y = 0`). Elegir `a` y `b` válidos es, visualmente, **agarrar un punto por encima del eje y otro por debajo**: si la curva es continua, para pasar de arriba a abajo tiene que cruzar el cero en el medio. Eso es Bolzano dibujado.

Receta gráfica antes de bisecar:
1. Ploteá `f` en un rango amplio.
2. Dibujá el eje con `plt.axhline(0)`.
3. Mirá dónde cruza: ese es el "vecindario" de la raíz.
4. Elegí `a` y `b`, uno de cada lado del cruce.

```python
import numpy as np, matplotlib.pyplot as plt
f = lambda x: x**3 - x - 2
x = np.linspace(0, 3, 400)
plt.plot(x, f(x))
plt.axhline(0, color="gray", lw=1)             # el eje: y = 0
plt.scatter([1, 2], [f(1), f(2)], zorder=5)    # a=1 (abajo), b=2 (arriba)
plt.grid(True); plt.show()
```

> Mostrá `bolzano_grafico.png`: a la **izquierda** un intervalo válido (un punto arriba y otro abajo, la curva cruza en la raíz ≈ 1.52); a la **derecha** uno inválido (los dos puntos del mismo lado, sin garantía). Frase: *"elegir el intervalo no es adivinar: es mirar el gráfico y agarrar un punto de cada lado del cruce."*

---

## 7 · CIERRE Y TRANSICIÓN (1 min)

> *"Regla para todo el parcial: primero hacé que corra, después NO le creas hasta probarlo con un caso conocido. El que aprueba no es el que no comete errores: es el que los caza rápido. En el próximo bloque vamos rutina por rutina, y ya con este radar puesto."*

---

### Checklist para vos (presentador)
- [ ] Tener `demos_bloque1.py` abierto y probado antes de empezar.
- [ ] DEMO 1 corrido en vivo como apertura, sin spoilear.
- [ ] Diapo grande con "🔴 gritan vs ⚫ mienten".
- [ ] Tener a mano el caso del aula virtual (B columna → `ValueError`).
- [ ] Diapo grande con los 4 comandos.
- [ ] `bolzano_grafico.png` listo para proyectar en la parte teórica.
- [ ] Cronómetro para el "cazá el bug".
