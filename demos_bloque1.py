import numpy as np

def sep(t): print("\n" + "="*60 + f"\n{t}\n" + "="*60)

# =====================================================================
# DEMO 1 - LA BOMBA DE ENTEROS (el error que reprueba sin avisar)
# =====================================================================
# Sistema:  2x + 1y = 3
#            1x + 3y = 5   ->  solucion exacta: x=0.8 , y=1.4
sep("DEMO 1: dtype int -> truncacion silenciosa (Gauss)")

def gauss_2x2(C):
    C = C.copy()
    k = C[1,0]/C[0,0]
    C[1,0:] = C[1,0:] - k*C[0,0:]      # <- aca se trunca si C es int
    y = C[1,2]/C[1,1]
    x = (C[0,2] - C[0,1]*y)/C[0,0]
    return x, y

A = np.array([[2,1],[1,3]])
b = np.array([[3],[5]])

C_int   = np.hstack((A, b))                 # SIN astype -> int64
C_float = np.hstack((A, b)).astype(float)   # correcto

print("dtype de C_int  :", C_int.dtype, "  <- la senal de alarma")
print("respuesta MAL   :", gauss_2x2(C_int),   "(corrio sin error)")
print("respuesta BIEN  :", gauss_2x2(C_float), "(exacta: 0.8 , 1.4)")

# =====================================================================
# DEMO 2 - COLUMNA vs FILA (tu ejemplo de Jacobi)
# =====================================================================
sep("DEMO 2: vector columna (n,1) vs fila (n,) -> norma fantasma")

X = np.array([1.0, 2.0, 3.0])        # (3,)   fila
P_fila = np.array([1.1, 1.9, 3.05])  # (3,)   fila  -> correcto
P_col  = P_fila.reshape(-1, 1)       # (3,1)  columna -> el bug clasico

print("X - P_fila  shape:", (X - P_fila).shape,
      " norma:", round(float(np.linalg.norm(X - P_fila)), 4))
print("X - P_col   shape:", (X - P_col).shape,
      " norma:", round(float(np.linalg.norm(X - P_col)), 4))
print(">> Con la columna, X-P se 'broadcastea' a (3,3): la norma "
      "que mide convergencia queda inflada y falsa, y NO tira error.")

# =====================================================================
# DEMO 3 - ALIASING: olvidar .copy()
# =====================================================================
sep("DEMO 3: P = X  vs  P = X.copy()")

X = np.array([10.0, 20.0])
P_alias = X          # mismo objeto en memoria
P_copy  = X.copy()   # objeto nuevo
X[0] = 999           # modifico X

print("con P = X       -> P =", P_alias, " (se contamino solo)")
print("con P = X.copy()-> P =", P_copy,  " (quedo intacto)")
print(">> En un bucle iterativo, P=X hace que err=||X-P||=0 en la "
      "1ra vuelta: corta al toque y devuelve un resultado prematuro.")

# =====================================================================
# DEMO 4 - ** vs ^   y   grados vs radianes
# =====================================================================
sep("DEMO 4: ** vs ^  y  radianes")

print("2 ** 3 =", 2**3, " (potencia, lo que queres)")
print("2 ^  3 =", 2^3,  " (XOR de enteros: da 1, corre sin quejarse)")
try:
    _ = 2.0 ^ 3          # con float SI explota
except TypeError as e:
    print("2.0 ^ 3 ->", type(e).__name__, ":", e)

print("np.sin(30)        =", round(float(np.sin(30)), 4),
      " (30 RADIANES, casi seguro NO es lo que queres)")
print("np.sin(np.radians(30)) =", round(float(np.sin(np.radians(30))), 4),
      " (30 grados = 0.5, correcto)")

# =====================================================================
# DEMO 5 - Simpson con n que no corresponde -> IndexError (ruidoso)
# =====================================================================
sep("DEMO 5: Simpson 1/3 con n impar -> IndexError")

def simpson13(y, h, n):
    A, k = 0.0, 0
    while k < n:
        A += (h/3)*(y[k] + 4*y[k+1] + y[k+2])
        k += 2
    return A

y = np.linspace(0, 1, 6)  # 6 puntos -> n = 5 (IMPAR: mal para 1/3)
try:
    simpson13(y, 0.2, 5)
except IndexError as e:
    print("n=5 (impar) ->", type(e).__name__, ":", e)
print(">> Este es un error RUIDOSO (Python avisa). El peligroso seria "
      "elegir n par pero equivocado: ahi da un area mal SIN avisar.")

# =====================================================================
# DEMO 6 - Los 2 comandos que salvan examenes
# =====================================================================
sep("DEMO 6: .shape y .dtype como diagnostico")

M = np.array([[1,2],[3,4]])
v = np.array([[5],[6]])
print("M.dtype =", M.dtype, "| M.shape =", M.shape)
print("v.shape =", v.shape, "<- columna: revisar antes de iterar")
print(">> Ante 'corre pero da mal': print(x.shape) y print(x.dtype). "
      "En 2 lineas encontras el 80% de los bugs silenciosos.")
