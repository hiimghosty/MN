# Guía de Métodos Numéricos — Declaraciones y Tips

> Las rutinas (001–027) te las dan en el examen y son **casi inmutables**. Esta guía
> no las repite: te dice **qué declarar**, **cómo rellenar los `[...]`** y dónde están
> las trampas. Pensada para tener al lado mientras resolvés.

---

## Setup de librerías

```python
import numpy as np
import jax.numpy as jnp
from jax import grad, jacfwd, hessian
import sympy as sp          # solo Taylor generalizado
import math                 # factorial en Taylor
```

- Usá `jnp` (no `np`) **dentro** de funciones que vayas a derivar con `grad`/`jacfwd`/`hessian`
  (Newton, Newton multidim, mínimos cuadrados). Si mezclás `np` con `grad`, falla.
- El resto (sistemas lineales, integración, EDOs) va con `np` normal.

---

## Qué significa `n` según el contexto

| Contexto | `n` es… |
|---|---|
| Métodos iterativos (ceros, Jacobi/Seidel) | tope de iteraciones |
| Integración | número de subintervalos |
| Datos tabulados | `len(x) - 1` |
| Interpolación | grado del polinomio = nº de puntos − 1 |
| EDO de un paso (020–023) | pasos desde `x0` → `xf = x0 + n·h` |
| EDO multipaso (024–027) | puntos **nuevos** tras `y3` → `xf = x0 + (3+n)·h` |

---

# 1 · Sistemas lineales `Ax = B` (001–003)

| Rutina | Declarar | Condición / tip |
|---|---|---|
| **001 Gauss** | `A`, `B`, `delta` (tolerancia ≈ cero) | Matriz ampliada: `C = np.hstack((A, B.reshape(-1,1)))`. Requiere `det(A) ≠ 0`. Trae pivoteo parcial incluido. |
| **002 Jacobi** | `A`, `B`, `P` (punto inicial), `tol`, `n` | `A` **diagonal dominante**. `X` se inicializa en ceros. |
| **003 Gauss-Seidel** | `A`, `B`, `P`, `tol`, `n` | `A` diagonal dominante. **`X = P.copy()`** (no ceros). Suele pedirse dominancia solo por intercambio de columnas. |

**Diagonal dominante:** `|aᵢᵢ| ≥ Σ|aᵢⱼ|` para cada fila. Si no lo es, **reordená filas o columnas**
antes de correr la rutina (si intercambiás columnas, recordá que reordenás también las incógnitas).

```python
C = np.hstack((A, B.reshape(-1, 1)))   # 001
n = A.shape[0]                         # orden del sistema
```

---

# 2 · Ceros de una función `f(x)=0` (004–008)

**Común a todas:** definí `f`, una tolerancia `tol` y un tope `n` de iteraciones.
Si el enunciado no da el intervalo/punto inicial, **graficá** `f` y elegí una zona que
contenga **una sola** raíz.

| Rutina | Declarar | Tip clave |
|---|---|---|
| **004 Bisección** | `a`, `b`, `c0=a` | `f(a)·f(b) < 0` obligatorio |
| **005 Falsa posición** | `a`, `b`, `c0=a` | igual que bisección |
| **006 Newton-Raphson** | `p0`, `f`, `df = grad(f)` | `p0` cercano a la raíz. `f` con `jnp` |
| **007 Punto fijo** | `p0`, `f`, `g` | `g` despejando una `x` de `f`; exige `|g'(p0)| < 1` |
| **008 Secante** | `x0`, `x1` | dos puntos, no necesita derivada |

```python
df = grad(f)        # 006: derivada automática (f debe usar jnp)
```

**Punto fijo — obtener `g`:** despejá una `x` de `f(x)=0`. Ej: `x²−2x+1=0 → x=√(2x−1)=g(x)`.
Puede haber varios despejes; quedate con el que cumpla `|g'(p0)|<1`.

---

# 3 · Sistemas no lineales (009 · Newton multidimensional)

```python
def sistema(var):
    x, y, *_ = var
    return np.array([f1(x, y), f2(x, y)])   # una entrada por ecuación

P0 = np.array([...], dtype=float)            # vector inicial (condicionado)
jacob_sist = jacfwd(sistema)                 # jacobiano automático
```

- **Optimización / Lagrange:** las soluciones son los ceros del gradiente, así que
  `sistema = grad(L)` y `jacob_sist = hessian(L)`. Lo demás igual.

---

# 4 · Mínimos cuadrados

Idea: minimizar el error cuadrático ⇒ buscar el cero del gradiente con la rutina 009.

```python
def f(var):
    a, b = var                      # coeficientes del modelo (ejemplo)
    model = b / (x_dat + a)         # tu modelo aquí
    return jnp.sum((model - y_dat) ** 2)

sistema    = grad(f)                # → rutina 009
jacob_sist = hessian(f)
# P[0], P[1], ... son los coeficientes ajustados
```

**Métricas (010):**
```python
y_pred = P[1] / (x_dat + P[0])
SST = jnp.sum((y_dat - jnp.mean(y_dat))**2)
SSR = jnp.sum((y_dat - y_pred)**2)
R2  = 1 - SSR/SST
```

---

# 5 · Interpolación (011–013)

Buscan el polinomio de grado `n−1` que pasa por los puntos `(x, y)`.

| Rutina | Inicializaciones |
|---|---|
| **011 Directa (Vandermonde)** | `m = len(x)`; luego `v = np.vander(x, N=m, increasing=True)`, `a = np.linalg.solve(v, y)` |
| **012 Lagrange** | `n = len(x)`, `P = np.poly1d([0])` |
| **013 Newton** | `p = np.poly1d([1, -x[0]])`, `a = 0`, `P = np.poly1d([y[0]])` |

> En 013 `a=0` es solo para reservar memoria; cualquier valor sirve.

---

# 6 · Integración (014–016)

```python
h = (b - a) / n
y = f(np.linspace(a, b, n + 1))     # función explícita
```

| Rutina | Restricción de `n` | `s` |
|---|---|---|
| **014 Trapecio** | cualquiera | 1 |
| **015 Simpson 1/3** | **par** (`n = 2s`) | 2 |
| **016 Simpson 3/8** | **múltiplo de 3** (`n = 3s`) | 3 |

**Datos tabulados:** no hay `f`; usá `n = len(x) - 1` y dentro del bucle `h = x[k+1] - x[k]`.

---

# 7 · Derivación numérica (017–019)

Son fórmulas de reemplazo directo (progresivas, centradas, regresivas) a 1.°–4.° orden de error.

- **Tabulados:** `f(x) → f[i]`. En el denominador usá el paso real (`x[i+1] - x[i]`), no `1`.
- **Centradas** necesitan puntos a ambos lados ⇒ no valen en los bordes de la tabla.
- Elegí progresiva en el borde izquierdo, regresiva en el derecho, centrada en el medio.

---

# 8 · EDOs (020–027) — el núcleo

Todas resuelven la EDO en **forma matricial de estado**: `y' = A(x)·y + B(x)`.

## 8.1 · Pasar de la EDO a `A` y `B`

Para una EDO **lineal** de orden `m`, despejada en la derivada mayor:

```
y^(m) = c0(x)·y + c1(x)·y' + … + c_(m-1)(x)·y^(m-1) + g(x)
```

El vector de estado y el bloque de “corrimiento” de derivadas:

```python
y0 = np.array([[y(x0)], [y'(x0)], …, [y^(m-1)(x0)]], dtype=float)  # m filas
m  = len(y0)
a  = np.hstack((np.zeros((m-1,1)), np.eye(m-1)))     # mueve y'→y', y''→y''…
```

Rellenado de `A` y `B`:

```python
A = lambda x: np.vstack((a, np.array(([c0, c1, …, c_{m-1}]), dtype=float)))  # coefs en orden
B = lambda x: np.vstack((np.zeros((m-1,1)), np.array(([ g(x) ]), dtype=float)))  # término indep.
```

**Ejemplos rápidos**

```python
# 3.er orden:  y''' = -5y + 4y' - 2y'' + sin(x)
A = lambda x: np.vstack((a, np.array(([-5, 4, -2]), dtype=float)))
B = lambda x: np.vstack((np.zeros((m-1,1)), np.array(([np.sin(x)]), dtype=float)))

# 1.er orden:  y' = k·y
A = lambda x: np.vstack((a, np.array(([k]), dtype=float)))
B = lambda x: np.vstack((np.zeros((m-1,1)), np.array(([0]), dtype=float)))
```

> Orden de los coeficientes en `A`: **`y, y', y'', …`** (de la derivada menor a la mayor).
> Para 1.er orden `a` queda vacío y `A` es solo `[[c0]]`.

## 8.2 · ⚠️ EDO NO LINEAL — 

Si en la derivada mayor aparece `y²`, `√y`, `y·y'`, `sin(y)`, `1/y`, etc., **no se puede
poner en `A`** (no hay coeficiente lineal). La técnica es:

1. La fila de `A` va en **cero** (no hay parte lineal).
2. **Todo** el lado derecho va en `B`, y `B` pasa a depender de `y`: **`B = lambda x, y:`**.
3. Sacá el escalar con `np.ravel(y)[0]`.
4. **Hay que tocar las llamadas a `B`** en la rutina: pasarles el `y` correspondiente.

```python
def f(H):                       # lado derecho, depende de la variable dependiente
    return ...

A = lambda x:    np.vstack((a, np.array(([0.0]), dtype=float)))                       # fila en 0
B = lambda x, y: np.vstack((np.zeros((m-1,1)), np.array(([ f(np.ravel(y)[0]) ]), dtype=float)))
```

Cambios en las llamadas (único retoque a la rutina “inmutable”):

```python
# RK04 (023): las 4 etapas
dy0 = A(x)@y0 + B(x, y0);  dy1 = A(x)@y1 + B(x, y1)
dy2 = A(x)@y2 + B(x, y2);  dy3 = A(x)@y3 + B(x, y3)

# Multipaso (024–027)
dy1 = A(x1)@y1 + B(x1, y1);  dy2 = A(x2)@y2 + B(x2, y2);  dy3 = A(x3)@y3 + B(x3, y3)
dy4 = A(x4)@p  + B(x4, p)        # o B(x4, mod) en los modificados
```

**Ejemplo trabajado — vaciado de tanque (Torricelli).**
EDO: `dH/dt = -(C·S·√(2gH)) / A_sup(H)`, donde `A_sup(H)` es el **área de la superficie
libre** (corte plano del líquido, no una superficie curva), que sale de un balance de volumen
`A_sup(H)·dH/dt = -Q_sal`.

```python
def dHdt(H):
    return -C*S*np.sqrt(2*g*H) / Asup(H)     # Asup: cilindro = πR²;  esfera = π(2RH - H²)

A = lambda x:    np.vstack((a, np.array(([0.0]), dtype=float)))
B = lambda x, y: np.vstack((np.zeros((m-1,1)), np.array(([dHdt(np.ravel(y)[0])]), dtype=float)))
```

> Como la EDO es **autónoma** (no depende explícito de `x`/`t`), el valor de `x` que recibe `B`
> ni se usa — pero se deja por simetría.

## 8.3 · Elegir `n` y `h`

```python
# Un paso (Euler, Heun, RK04):
n = int((xf - x0) / h)

# Multipaso (necesita y0,y1,y2,y3 → ya cubriste 3·h al arrancar):
n = int((xf - x0) / h) - 3
```

**Cuidado con el dominio:** si la solución toca un punto donde la EDO se rompe
(`√` de negativo, denominador `→ 0`), elegí `n` para **no cruzarlo**. En el tanque, por ejemplo,
no pasar de `H = 0`.

## 8.4 · Valores iniciales para multipaso (RK04 de arranque)

Milne/Hamming necesitan 4 puntos consecutivos. Lo típico: correr **RK04 tres pasos** para
generar `y1, y2, y3`, y recién ahí entrar al multipaso.

```python
x1 = x0 + h;  x2 = x1 + h;  x3 = x2 + h    # las x de y1, y2, y3
```

**Patrón para tabular** (opcional pero cómodo en el examen): una matriz con columnas
`[iteración, x, y…]`.

```python
resultados = np.zeros((n_milne + n_rk04 + 1, 2 + m))
resultados[0, :] = np.concatenate(([0, x0], y0))
# … RK04 llena filas 1..3,  multipaso llena 4..
resultados[i+1, :] = np.concatenate(([i+1, x], y0.flatten()))
```

## 8.5 · Diferencias entre métodos (lo que cambia, no la rutina entera)

| Rutina | Tipo | Lo que la distingue |
|---|---|---|
| **020 Euler** | un paso | `y = y0 + h·dy` |
| **021 Taylor** | un paso (escalar) | **no usa `A`/`B`**; declarás `dy(x,y)` y derivadas por regla de la cadena total |
| **022 Heun** | un paso | promedio de dos pendientes |
| **023 RK04** | un paso | 4 pendientes a medio paso; suele generar el arranque de los multipaso |
| **024 Milne-Simpson** | multipaso | corrector Simpson: `y = y2 + (h/3)(dy2 + 4·dy3 + dy4)` |
| **025 Milne mod.** | multipaso | + modificador `mod = p + (28/29)(y3 − p0)`; **`p0` no se declara** (se crea con `if i>0`) |
| **026 Hamming** | multipaso | corrector: `y = (9y3 − y1)/8 + (3/8)h(−dy2 + 2·dy3 + dy4)` |
| **027 Hamming mod.** | multipaso | + modificador `mod = p + (112/121)(y3 − p0)`; `p0` no se declara |

> El **predictor** (Milne) es igual en todos: `p = y0 + (4/3)h(2·dy1 − dy2 + 2·dy3)`.
> Lo que cambia entre Milne y Hamming es la **línea correctora**.

**Taylor (021):** para `y' = f(x,y)` escalar. Declarás `dy` y las superiores con la cadena total:
```python
def dy(x, y):  return ...
def d2y(x, y): return grad(dy, 0)(x, y) + grad(dy, 1)(x, y) * dy(x, y)
# d3y, d4y… análogo, derivando la anterior
```

---

# 9 · Checklist de examen / trampas comunes

- [ ] **¿La EDO es lineal?** Si tiene `y²`, `√y`, `y·y'`, `sin(y)`, `1/y`… → fila de `A` en `0`,
      todo a `B = lambda x, y:` y pasá `y` en cada llamada a `B`.
- [ ] **Orden de coeficientes en `A`:** `y, y', y'', …` (menor a mayor derivada).
- [ ] **Vector inicial `y0`:** una fila por cada condición inicial (orden `m` → `m` filas).
- [ ] **Diagonal dominante** antes de Jacobi/Seidel; Seidel arranca con `X = P.copy()`.
- [ ] **Simpson:** `n` par (1/3) o múltiplo de 3 (3/8).
- [ ] **Ceros:** un intervalo con **una sola** raíz; `f(a)·f(b)<0` para bisección/falsa posición.
- [ ] **`jnp` (no `np`)** dentro de funciones que derivás con `grad`/`jacfwd`/`hessian`.
- [ ] **Multipaso:** 4 puntos de arranque (RK04); `n = int((xf−x0)/h) − 3`.
- [ ] **Dominio:** no iterar más allá de donde la EDO se rompe (`√` de negativo, denominador `→ 0`).
- [ ] **`np.ravel(y)[0]`** para sacar el escalar en EDOs no lineales (evita líos de forma `(1,)` vs `(1,1)`).

---

# 10 · Tabla maestra

| #   | Rutina | Declarar mínimo |
|-----|--------|-----------------|
| 001 | Gauss | `A`, `B`, `delta` |
| 002 | Jacobi | `A`, `B`, `P`, `tol`, `n` |
| 003 | Gauss-Seidel | `A`, `B`, `P`(=X init), `tol`, `n` |
| 004 | Bisección | `a`, `b`, `c0=a`, `tol`, `n` |
| 005 | Falsa posición | `a`, `b`, `c0=a`, `tol`, `n` |
| 006 | Newton-Raphson | `p0`, `f`, `df=grad(f)`, `tol`, `n` |
| 007 | Punto fijo | `p0`, `f`, `g`, `tol`, `n` |
| 008 | Secante | `x0`, `x1`, `tol`, `n` |
| 009 | Newton multidim | `sistema`, `jacob_sist=jacfwd(sistema)`, `P0`, `tol`, `m` |
| 010 | Correlación R² | `y_data`, `y_pred` |
| 011 | Interp. directa | `x`, `y`, `m=len(x)` |
| 012 | Lagrange | `x`, `y`, `n=len(x)`, `P=poly1d([0])` |
| 013 | Newton interp. | `x`, `y`, `p`, `a=0`, `P=poly1d([y[0]])` |
| 014 | Trapecio | `a`,`b`,`n`,`f` (o `x`,`y`) |
| 015 | Simpson 1/3 | idem, `n` par |
| 016 | Simpson 3/8 | idem, `n` múltiplo de 3 |
| 017–019 | Derivadas | fórmula directa; tabulados con paso real |
| 020 | Euler | `x0`,`y0`,`h`,`n`,`A`,`B` |
| 021 | Taylor | `x0`,`y0`,`h`,`n`,`dy` y derivadas |
| 022 | Heun | `x0`,`y0`,`h`,`n`,`A`,`B` |
| 023 | RK04 | `x0`,`y0`,`h`,`n`,`A`,`B` |
| 024 | Milne-Simpson | `x0`,`h`,`n`,`y0..y3`,`A`,`B` |
| 025 | Milne mod. | idem (sin declarar `p0`) |
| 026 | Hamming | `x0`,`h`,`n`,`y0..y3`,`A`,`B` |
| 027 | Hamming mod. | idem (sin declarar `p0`) |