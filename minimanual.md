# Resumen de Rutinas – Métodos Numéricos - 

---

## Rutinas para Resolver Ax = B

### 001 – Gauss

Aplicamos Gauss directo. Declaración relevante (asumiendo que `B` es un array 1D):

```python
C = np.hstack((A, B.reshape(-1, 1)))
```

> `n`: tamaño del sistema (número de ecuaciones o filas de A)

---

### 002 – Jacobi

> `n`: tamaño del sistema (número de incógnitas)

---

### 003 – Gauss-Seidel

> `n`: tamaño del sistema (número de incógnitas)

---

## Rutinas para Hallar Ceros

### 004 – Bisección

- Declarar límites `a`, `b`
- `c0 = a`

> `n`: número máximo de iteraciones

---

### 005 – Falsa Posición

- Igual a Bisección

> `n`: número máximo de iteraciones

---

### 006 – Newton-Raphson

- Declarar:
  - `p0`
  - `f`
  - `df = grad(f)`

> También podría ser `f = grad(f)` según qué se maximiza.

> `n`: número máximo de iteraciones

---

### 007 – Punto Fijo

- Declarar:
  - `p0`
  - `f(x) = 0`
  - `g(x)` (despejando la x)

> `n`: número máximo de iteraciones

---

### 008 – Secante

- Declarar:
  - `x0`
  - `x1`

> `n`: número máximo de iteraciones

---

## Multidimensional

### 009 – Newton-Raphson Multidimensional

Si hay varias funciones:

```python
def sistema(var):
    x, y, ... = var
    return np.array([f1(x, y, ...), f2(x, y, ...), ...])
```

- `p0` se declara como vector

Jacobiano:

```python
jacob_sist = jacfwd(sistema)
```

**Caso Lagrange:**

- Se debe maximizar ⇒ raíces de:

```python
sistema = grad(L)
```

- Lo demás es igual

> `n`: número máximo de iteraciones

---

## Mínimos Cuadrados

**Datos:** `x_dat`, `y_dat`

**Idea clave:** minimizar el error cuadrático entre modelo y datos.

Definición correcta:

```python
def f(var):
    # var = coeficientes del modelo
    # ejemplo: var = [a, b]

    model = var[1] / (x_dat + var[0])  # ejemplo de modelo

    return jnp.sum((model - y_dat) ** 2)
```

Derivadas:

```python
sistema = grad(f)
jacob_sist = hessian(f)
```

Predicción y métricas:

```python
y_pred = P[1] / (x_dat + P[0])

SST = jnp.sum((y_dat - jnp.mean(y_dat)) ** 2)
SSR = jnp.sum((y_dat - y_pred) ** 2)

R = 1 - SSR / SST
```

> `n`: cantidad de datos (`len(x_dat)`)

---

## Interpolación

### 011 – Interpolación Directa

```python
m = len(x)
```

> `n`: cantidad de puntos − 1 (grado del polinomio = n)

---

### 012 – Lagrange

```python
n = len(x)
P = np.poly1d([0])
```

> `n`: cantidad de puntos

---

### 013 – Newton

```python
p=np.poly1d([1,-x[0]])
a=0 #Puede ser cualquier valor, con tal de crear un espacio de memoria para "a"
P=np.poly1d([y[0]])
```

> `n`: cantidad de puntos

---

## Rutinas de Integración

### 014 – Trapecio Compuesto

Declarar: `a`, `b`, `n`, `f` (función explícita) o `x`, `y` (tabulado con `n = len(x) - 1`)

```python
h = (b - a) / n
# o
h = x[1] - x[0]
```

- `s = 1`

> `n`: número de subintervalos

---

### 015 – Simpson 1/3

Declarar: `a`, `b`, `n` **par** — o `x`, `y` con `n = len(x) - 1` par.

```python
h = (b - a) / n
# o
h = x[1] - x[0]
```

- `s = 2`
- `n = 2m`

> `n`: número de subintervalos (debe ser par)

---

### 016 – Simpson 3/8

Declarar: `a`, `b`, `n` **múltiplo de 3** — o `x`, `y` con `n = len(x) - 1` múltiplo de 3.

```python
h = (b - a) / n
# o
h = x[1] - x[0]
```

- `s = 3`
- `n = 3m`

> `n`: número de subintervalos (múltiplo de 3)

---

## Derivación Numérica

### 017 – Progresivas (Forward)

- Primer y segundo orden
- En tabulados: `f(x)` → `f[i]`, `h = 1` (índice)
- En denominador usar delta real (ej: `x[i+1] - x[i]`)

> `n`: cantidad de datos

---

### 018 – Centrales

- Segundo y cuarto orden
- Requiere punto central (no válido en bordes)

> `n`: cantidad de datos

---

### 019 – Regresivas (Backward)

- Primer y segundo orden

> `n`: cantidad de datos

---
## Ecuaciones diferenciales ordinarias

### Declaraciones comunes – Rutinas 020–027

EDO lineal de orden `m`, despejada en su derivada mayor:

```
y^(m) = c0(x)·y + c1(x)·y' + ... + c_(m-1)(x)·y^(m-1) + g(x)
```

Declarar siempre:

```python
x0 = ...   # punto inicial
h  = ...   # tamaño de paso
n  = ...   # cantidad de iteraciones
```

Vector inicial (columna), con **tantas filas como el orden de la EDO**:

```python
y0 = np.array([
    [y(x0)],
    [y'(x0)],
    [y''(x0)],
    ...
], dtype=float)
m = len(y0)
```

> Para una EDO de orden `m` se necesitan exactamente `m` condiciones iniciales: `y(x0), y'(x0), ..., y^(m-1)(x0)`.

---

### Cómo rellenar `A` y `B`

```python
A = lambda x: np.vstack((a, np.array(([...]), dtype=float)))
B = lambda x: np.vstack((np.zeros((m-1, 1)), np.array(([...]), dtype=float)))
```

- En `[...]` de `A`: coeficientes de `y, y', y'', ..., y^(m-1)` (en ese orden)
- En `[...]` de `B`: término independiente `g(x)`

**Ejemplo – tercer orden:** `y''' = -5y + 4y' - 2y'' + sin(x)`

```python
y0 = np.array([[y_0], [dy_0], [d2y_0]], dtype=float)

A = lambda x: np.vstack((a, np.array(([-5, 4, -2]), dtype=float)))
B = lambda x: np.vstack((np.zeros((m-1, 1)), np.array(([np.sin(x)]), dtype=float)))
```

**Ejemplo – primer orden:** `y' = k·y`

```python
y0 = np.array([[y_0]], dtype=float)

A = lambda x: np.vstack((a, np.array(([k]), dtype=float)))
B = lambda x: np.vstack((np.zeros((m-1, 1)), np.array(([0]), dtype=float)))
```

> Si la EDO tiene términos no lineales como `y²`, `y·y'` o `sin(y)`, estas plantillas pueden no ser válidas directamente.

---

### 020 – Euler

- Método de un paso
- Declarar `x0`, `y0`, `h`, `n` y completar `A`, `B`
- `n`: pasos desde `x0` → último punto: `xf = x0 + n·h`

```python
n = int((xf - x0) / h)
```

---

### 021 – Taylor

- Para EDO escalar de primer orden: `y' = f(x, y)`
- **No usa matrices `A` ni `B`**
- Declarar `dy(x, y)` y las derivadas superiores mediante regla de la cadena total

```python
def dy(x, y):
    return ...
```

- En la versión generalizada, distinguir:

```python
orden_taylor = ...  # cantidad de términos
n_pasos      = ...  # avances en x
```

---

### 022 – Heun

- Método de un paso
- Mismas declaraciones que Euler: `x0`, `y0`, `h`, `n` y completar `A`, `B`

```python
n = int((xf - x0) / h)
```

---

### 023 – Runge-Kutta orden 4 (RK04)

- Método de un paso
- Mismas declaraciones que Euler y Heun
- Frecuentemente usado para generar los valores iniciales de los métodos multipaso

```python
n = int((xf - x0) / h)
```

---

### Valores iniciales para métodos multipaso (024–027)

Requieren cuatro vectores iniciales consecutivos, normalmente generados con RK04:

```python
y0  # en x0
y1  # en x0 + h
y2  # en x0 + 2h
y3  # en x0 + 3h

x1 = x0 + h
x2 = x1 + h
x3 = x2 + h
```

- `n`: puntos **nuevos** calculados después de `y3`
- Último punto: `xf = x0 + (3 + n)·h`

```python
n = int((xf - x0) / h) - 3
```

---

### 024 – Milne-Simpson

- Método multipaso predictor-corrector
- Declarar `x0`, `h`, `n`, `y0`, `y1`, `y2`, `y3` y completar `A`, `B`

---

### 025 – Milne-Simpson modificado

- Igual que 024
- `p0` **no se declara** antes del bucle; la rutina lo guarda al final de la primera iteración

---

### 026 – Hamming

- Método multipaso predictor-corrector
- Declarar `x0`, `h`, `n`, `y0`, `y1`, `y2`, `y3` y completar `A`, `B`

---

### 027 – Hamming modificado

- Igual que 026
- `p0` **no se declara** antes del bucle; la rutina lo guarda al final de la primera iteración

---

### Resumen rápido

| Rutina                         | Tipo      | Valores iniciales        | `n` representa                |
|-------------------------------|-----------|--------------------------|-------------------------------|
| 020 – Euler                   | Un paso   | `y0`                     | Pasos desde `x0`              |
| 021 – Taylor                  | Un paso   | `y0`                     | Pasos desde `x0`              |
| 022 – Heun                    | Un paso   | `y0`                     | Pasos desde `x0`              |
| 023 – RK04                    | Un paso   | `y0`                     | Pasos desde `x0`              |
| 024 – Milne-Simpson           | Multipaso | `y0`, `y1`, `y2`, `y3`   | Puntos nuevos después de `y3` |
| 025 – Milne-Simpson mod.      | Multipaso | `y0`, `y1`, `y2`, `y3`   | Puntos nuevos después de `y3` |
| 026 – Hamming                 | Multipaso | `y0`, `y1`, `y2`, `y3`   | Puntos nuevos después de `y3` |
| 027 – Hamming mod.            | Multipaso | `y0`, `y1`, `y2`, `y3`   | Puntos nuevos después de `y3` |

## Observacion importante sobre `n`

| Contexto | Significado de `n` |
|---|---|
| Métodos iterativos | Número de iteraciones |
| Integración numérica | Número de subintervalos |
| Datos tabulados | `len(x) - 1` |
| Interpolación | Grado del polinomio |