# File: proyecto_final.py
# Date: 23/04/2024

import numpy as np
import matplotlib.pyplot as plt

# Parámetros
N = 50  # Número de partículas
L = 1.0  # Longitud del contenedor
T = 1.0   # Temperatura
V = L**3  # Volumen del contenedor

# Inicialización de posiciones aleatorias dentro del contenedor
posiciones_iniciales = L * np.random.rand(N, 3)
posiciones_finales = posiciones_iniciales.copy()

# Función para calcular la energía total del sistema
def calcular_energia_total(posiciones):
    energia_total = 0.0
    for i in range(N):
        for j in range(i+1, N):
            distancia = np.linalg.norm(posiciones[i] - posiciones[j])
            energia_total += 4 * (1/distancia**12 - 1/distancia**6)
    return energia_total

# Función para simular el sistema utilizando el método de Montecarlo
def montecarlo(posiciones, temperatura, num_pasos):
    for paso in range(num_pasos):
        # Seleccionar una partícula aleatoria
        indice = np.random.randint(N)
        # Calcular la energía del sistema antes del movimiento
        energia_anterior = calcular_energia_total(posiciones)
        # Generar un movimiento aleatorio para la partícula seleccionada
        nueva_posicion = posiciones[indice] + 0.1 * (np.random.rand(3) - 0.5)
        # Calcular la energía del sistema después del movimiento
        energia_nueva = calcular_energia_total(posiciones)
        # Calcular el cambio en la energía
        delta_energia = energia_nueva - energia_anterior
        # Aceptar o rechazar el movimiento según el criterio de Metropolis
        if delta_energia < 0 or np.random.rand() < np.exp(-delta_energia / temperatura):
            posiciones[indice] = nueva_posicion
    return posiciones

# Ejecutar la simulación
num_pasos = 10000
posiciones_finales = montecarlo(posiciones_finales, T, num_pasos)

# Graficar las posiciones iniciales y finales de las partículas
fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title('Posiciones Iniciales')
ax1.scatter(posiciones_iniciales[:, 0], posiciones_iniciales[:, 1], posiciones_iniciales[:, 2])

ax2 = fig.add_subplot(122, projection='3d')
ax2.set_title('Posiciones Finales')
ax2.scatter(posiciones_finales[:, 0], posiciones_finales[:, 1], posiciones_finales[:, 2])

plt.show()