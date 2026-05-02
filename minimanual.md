# Resumen de Rutinas – Métodos Numéricos - EST. Mauricio Benitez

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
n = len(x)
P = np.poly1d([y[0]])
p = np.poly1d([1])
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

## Observacion importante sobre `n`

| Contexto | Significado de `n` |
|---|---|
| Métodos iterativos | Número de iteraciones |
| Integración numérica | Número de subintervalos |
| Datos tabulados | `len(x) - 1` |
| Interpolación | Grado del polinomio |