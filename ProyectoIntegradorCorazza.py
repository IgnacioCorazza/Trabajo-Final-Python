# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import betaprime

# Se define la función de pulso
def pulso(t, t0, sigma, omega, epsilon=1e-10): # Se agrega epsilon para evitar división por cero
    gauss = np.exp(-(t - t0)**2 / (2 * sigma**2))
    seno = np.cos(omega * (t - t0 + sigma))
    S = gauss * seno
    A = 1 / (np.trapezoid(S, t) + epsilon)  
    return A * S

# Se simula una medición
def simular_medicion(t, t0s, probs, sigma0=0.05, sigma_sigma=0.005, omega0=5*np.pi, sigma_omega=0.05):
    señal = np.zeros_like(t)
    for t0, p in zip(t0s, probs):
        if np.random.rand() < p:
            x = betaprime.rvs(a=4, b=2)
            t_centro = t0 + (x - 4 / 2) * 0.15
            sigma = np.random.normal(sigma0, sigma_sigma)
            omega = np.random.normal(omega0, sigma_omega)
            señal += pulso(t, t_centro, sigma, omega)
    ruido_rel = np.random.normal(1, 0.1, size=t.shape)
    ruido_abs = np.random.normal(0, 0.2, size=t.shape)
    fondo = (-t + 2) / 100
    return señal * ruido_rel + ruido_abs + fondo

# Rutina para simular N mediciones
def simular_N_mediciones(N, t, t0s, probs):
    return np.array([simular_medicion(t, t0s, probs) for _ in range(N)])

# Tiempo y parámetros de la simulación
t = np.linspace(0, 8, 4000)
t0s = [1.8, 2.8, 5]
probs = [0.3, 0.1, 0.6]

# Simular animación: 50 mediciones
mediciones = simular_N_mediciones(50, t, t0s, probs)

# Crear animación
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
background = ax.fill_between(t, -3, 18, color='white', alpha=1)
ax.set_xlim(0, 8)
ax.set_ylim(-3, 18)
ax.set_xlabel('tiempo (μs)')
ax.set_ylabel('Tensión (mV)')
ax.set_title('Simulación de medición')
line, = ax.plot([], [], lw=1, color='blue')  
def update(frame):
    line.set_data(t, mediciones[frame])
    return line,

ani = FuncAnimation(fig, update, frames=len(mediciones), blit=True)
ani.save(r'D:\Nacho\IB\2025\Maestría\Python\ib25-deteccion\simulacion_medicion.gif', fps=5)

# %%
# Número de mediciones a promediar
N = 100000

# Simular N mediciones
mediciones = simular_N_mediciones(N, t, t0s, probs)

# Calcular el promedio de las mediciones
promedio = np.mean(mediciones, axis=0)

# Graficar el promedio
plt.figure(figsize=(10, 6))
plt.plot(t, promedio, lw=2, color='blue')
plt.xlabel('tiempo (μs)')
plt.ylabel('Tensión (mV)')
plt.gca().set_facecolor('white')
plt.title(f'Promedio de {N} mediciones simuladas')
plt.grid(True, color='black')
plt.gca().spines['bottom'].set_color('white')
plt.gca().spines['top'].set_color('white')
plt.gca().spines['right'].set_color('white')
plt.gca().spines['left'].set_color('white')
plt.tight_layout()
plt.savefig(r'D:\Nacho\IB\2025\Maestría\Python\ib25-deteccion\promedio_medicion.png', dpi=300)
plt.show()


# %% [markdown]
# 3.  Obtenga los datos de promedios de 10000 mediciones para tres casos
#     diferentes
# 
#     -   Los picos en $t_{0}=1.8, 2.8, 5$ tienen probabilidades de
#         ocurrir $P =  1, 0, 0$
#     -   Los picos en $t_{0}=1.8, 2.8, 5$ tienen probabilidades de
#         ocurrir $P = 0.8, 0.1, 0.1$
#     -   Los picos en $t_{0}=1.8, 2.8, 3.5$ tienen probabilidades de
#         ocurrir $P = 0.85, 0.1, 0.05$
# 
#     Grafique los tres casos, indicando en la figura la posición del
#     máximo para cada pico y el ancho a altura mitad.

# %%
from scipy.signal import find_peaks, peak_widths

fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Casos definidos
casos = [
    {"t0s": [1.8, 2.8, 5],    "probs": [1, 0, 0],    "titulo": "Caso 1"},
    {"t0s": [1.8, 2.8, 5],    "probs": [0.8, 0.1, 0.1], "titulo": "Caso 2"},
    {"t0s": [1.8, 2.8, 3.5],  "probs": [0.85, 0.1, 0.05], "titulo": "Caso 3"},
]

for i, caso in enumerate(casos):
    mediciones = simular_N_mediciones(10000, t, caso["t0s"], caso["probs"])
    promedio = mediciones.mean(axis=0)

    # Parámetros de detección
    distancia_pts = int(1 / (t[1] - t[0]))  # Separación mínima entre picos (1 μs)
    altura_min = 0.15  # Altura mínima para evitar ruido

    # Detectar picos
    peaks, props = find_peaks(promedio, height=altura_min, distance=distancia_pts)

    # Ordenar por altura descendente, mayor a 3 picos
    if len(peaks) > 3:
        top_idx = np.argsort(props["peak_heights"])[-3:]
        peaks = peaks[top_idx]

    # Calcular FWHM
    results_half = peak_widths(promedio, peaks, rel_height=0.5)

    # Gráfica
    axs[i].plot(t, promedio, label='Promedio de 10,000 mediciones')
    axs[i].set_title(caso["titulo"])
    axs[i].set_ylabel('Tensión (mV)')
    axs[i].set_xlabel('Tiempo (μs)')
    axs[i].tick_params(axis='x', labelsize=10)
    axs[i].grid(True)

    for j, idx in enumerate(peaks):
        fwhm = results_half[0][j] * (t[1] - t[0])
        axs[i].plot(t[idx], promedio[idx], 'rx')
        axs[i].annotate(f'Máx: {t[idx]:.2f} μs\n ancho ≈ {fwhm:.2f} μs',
                        xy=(t[idx], promedio[idx]),
                        xytext=(t[idx] + 0.2, promedio[idx] + 0.5),
                        arrowprops=dict(arrowstyle='->'))

axs[-1].set_xlabel('tiempo (μs)')
plt.tight_layout()
plt.show()


# %% [markdown]
# 4.  Utilizando los datos del inciso anterior, suponiendo que los tres
#     picos tienen la misma forma, obtenga una estimación del área de cada
#     pico en los tres casos mediante un ajuste por cuadrados mínimos.
#     Para ello utilice como modelo una función que devuelve los valores
#     del pico medido en el primer punto donde no hay interferencias de
#     otros picos (con probabilidades $P =  1, 0, 0$). Para ello cree una
#     función que, para cada valor de $t$ obtenga el valor correspondiente
#     interpolando la función "medida", adecuadamente parametrizada para
#     trasladarla y escalarla.

# %%
import numpy as np
from scipy.signal import find_peaks, peak_widths

# Para cada caso, se simulan las mediciones, se detectan picos, y se calcula el área bajo cada pico.
for i, caso in enumerate(casos):
    mediciones = simular_N_mediciones(10000, t, caso["t0s"], caso["probs"])
    promedio = mediciones.mean(axis=0)

    # Parámetros de detección
    distancia_pts = int(1 / (t[1] - t[0]))  # Separación mínima entre picos (1 μs)
    altura_min = 0.15  # Altura mínima para evitar ruido

    # Detectar picos
    peaks, props = find_peaks(promedio, height=altura_min, distance=distancia_pts)

    # Ajuste para detectar el número de picos 
    if len(peaks) < 3:
        # Se reduce la altura mínima para el Caso 2 y 3
        if caso["titulo"] == "Caso 2":
            altura_min = 0.1  # Reducir altura mínima para Caso 2
        elif caso["titulo"] == "Caso 3":
            altura_min = 0.05  # Reducir aún más para Caso 3
        
        # Recalcular picos con nueva altura mínima
        peaks, props = find_peaks(promedio, height=altura_min, distance=distancia_pts)

    # Ordenar por altura descendente, mayor a 3 picos
    if len(peaks) > 3:
        top_idx = np.argsort(props["peak_heights"])[-3:]
        peaks = peaks[top_idx]

    # Calcular FWHM
    results_half = peak_widths(promedio, peaks, rel_height=0.5)

    # Imprimir resultados de los picos detectados
    for j, idx in enumerate(peaks):
        fwhm = results_half[0][j] * (t[1] - t[0])

        # Verificar si el ancho es mayor que un umbral mínimo
        if fwhm > 0.1:  # Umbral de 0.1 μs para evitar anchos de pico demasiado pequeños
            # Cálculo del área bajo el pico
            # Encontrar los índices alrededor del pico donde la señal cruza el eje X
            start_idx = np.argmax(promedio[:idx] < 0)  # Primer cruce hacia abajo antes del pico
            end_idx = np.argmax(promedio[idx:] < 0) + idx  # Primer cruce hacia abajo después del pico

            # Si no se encuentran cruces, se usa el inicio y el final del arreglo
            if start_idx == 0: start_idx = idx - int(fwhm * 5)  # Ajustar el rango de inicio
            if end_idx == 0: end_idx = idx + int(fwhm * 5)  # Ajustar el rango de fin

            # Integrar el área usando la regla del trapecio
            area = np.trapezoid(promedio[start_idx:end_idx], t[start_idx:end_idx])

            # Imprimir resultados
            print(f"{caso['titulo']} - Pico en {t[idx]:.2f} μs:")
            print(f"  FWHM ≈ {fwhm:.2f} μs")
            print(f"  Área bajo el pico: {area:.2f} mV·μs\n")
+

# %%
import numpy as np
from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt

# Definición de la función de pulso y simulación de mediciones
t = np.linspace(0, 8, 4000)

# Casos 
casos = [
    {"t0s": [1.8, 2.8, 5],    "probs": [1, 0, 0],    "titulo": "Caso 1"},
    {"t0s": [1.8, 2.8, 5],    "probs": [0.8, 0.1, 0.1], "titulo": "Caso 2"},
    {"t0s": [1.8, 2.8, 3.5],  "probs": [0.85, 0.1, 0.05], "titulo": "Caso 3"},
]

for i, caso in enumerate(casos):
    mediciones = simular_N_mediciones(10000, t, caso["t0s"], caso["probs"])
    promedio = mediciones.mean(axis=0)

    # Parámetros de detección
    distancia_pts = int(1 / (t[1] - t[0]))  # Separación mínima entre picos (1 μs)
    altura_min = 0.15  # Altura mínima para evitar ruido

    # Detectar picos
    peaks, props = find_peaks(promedio, height=altura_min, distance=distancia_pts)

    # Calcular FWHM
    results_half = peak_widths(promedio, peaks, rel_height=0.5)

    print(f"\n==== {caso['titulo']} ====")
    for j, idx in enumerate(peaks):
        fwhm = results_half[0][j] * (t[1] - t[0])
        start_idx = int(results_half[2][j])
        end_idx = int(results_half[3][j])
        # Asegurarse de que los índices están en el rango válido
        start_idx = max(0, min(start_idx, len(t) - 1))
        end_idx = max(0, min(end_idx, len(t) - 1))
        area = np.trapezoid(promedio[start_idx:end_idx], t[start_idx:end_idx])
        print(f"Pico {j+1}: t = {t[idx]:.3f} μs | FWHM = {fwhm:.3f} μs | Área ≈ {area:.3f}")


# %%



