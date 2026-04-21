import jax.numpy as jnp
from jax import config, grad

config.update("jax_enable_x64", True)


# Datos tabulados del enunciado
t = jnp.array([0.03, 0.0305, 0.0310, 0.0315, 0.0320], dtype=jnp.float64)
I_tab = jnp.array(
    [19.98866, 19.989105, 19.989532, 19.989943, 19.990337], dtype=jnp.float64
)

# Punto de interés: t = 0.031 está en la posición 2
x = 2
t0 = t[x]
h = t[1] - t[0]

# Derivada progresiva de primer orden
dI_dt_estimada = (I_tab[x + 1] - I_tab[x]) / h
I_estimada = I_tab[x]


def E(I_prima, I_t):
    return 0.1 * I_prima + 8 * I_t


def I_real(t):
    return 20 - jnp.exp(-80 * t) / 8


# Derivada exacta de I(t)
dI_dt_real = grad(I_real)

# Valor estimado de E
E_estimada = E(dI_dt_estimada, I_estimada)

# Error en la derivada
error_derivada = jnp.abs(dI_dt_real(t0) - dI_dt_estimada)

# Valor exacto de E y error en el voltaje
E_real_valor = E(dI_dt_real(t0), I_real(t0))
error_voltaje = jnp.abs(E_real_valor - E_estimada)

print(f"a) Valor estimado de I'({t0:.4f}): {dI_dt_estimada:.6f}")
print(f"b) Valor estimado de E({t0:.4f}): {E_estimada:.6f}")
print(f"c) El error en la estimación de I'({t0:.4f}) es: {error_derivada:.6f}")
print(f"d) El error en la estimación de E({t0:.4f}) es: {error_voltaje:.6f}")
