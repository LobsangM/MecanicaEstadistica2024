import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Definir constantes
N = 500  # Número de partículas
V = 1    # Volumen (cúbico)
T = 1    # Temperatura (relativa)
dt = 0.01  # Incremento de tiempo
k_B = 1.38e-23  # Constante de Boltzmann en J/K
m = 1.67e-27  # Masa de una partícula en kg (por ejemplo, una molécula de hidrógeno)

# Función para generar velocidades aleatorias dentro del rango permitido usando Maxwell-Boltzmann
def generate_velocity_maxwell_boltzmann(T, N):
    sigma = np.sqrt(k_B * T / m)
    velocities = np.random.normal(0, sigma, (N, 3))
    return velocities

# Función para calcular las posiciones iniciales de las partículas
def generate_initial_positions(N):
    return np.random.uniform(0, 1, size=(N, 3))

# Función para simular el movimiento de las partículas
def simulate_particles(positions, velocities, dt, max_steps=1000):
    steps = 0

    while steps < max_steps:
        new_positions = positions + velocities * dt

        # Rebotar partículas en las paredes del volumen (simple caja cúbica)
        for i in range(len(new_positions)):
            for j in range(3):
                if new_positions[i, j] < 0 or new_positions[i, j] > 1:
                    velocities[i, j] = -velocities[i, j]  # Invertir la dirección de la velocidad

        positions = new_positions
        steps += 1

    return positions

# Graficar posiciones iniciales y finales
def plot_positions(initial_positions, final_positions):
    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(initial_positions[:,0], initial_positions[:,1], initial_positions[:,2])
    ax1.set_title('Posiciones Iniciales')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(final_positions[:,0], final_positions[:,1], final_positions[:,2])
    ax2.set_title('Posiciones Finales')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    plt.tight_layout()
    plt.show()

# Función para la distribución de Maxwell-Boltzmann
def maxwell_boltzmann(v, T, m):
    factor = (m / (2 * np.pi * k_B * T))**1.5
    return factor * 4 * np.pi * v**2 * np.exp(-m * v**2 / (2 * k_B * T))

# Graficar velocidades y realizar ajuste
def plot_velocities_with_fit(velocities):
    # Calcular las magnitudes de las velocidades
    speeds = np.linalg.norm(velocities, axis=1)
    
    # Graficar histograma
    hist, bins = np.histogram(speeds, bins=30, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Ajustar la distribución de Maxwell-Boltzmann
    popt, pcov = curve_fit(lambda v, T: maxwell_boltzmann(v, T, m), bin_centers, hist, p0=[T])

    # Calcular velocidad más probable y velocidad media
    v_p = np.sqrt(2 * k_B * T / m)
    v_mean = np.sqrt(8 * k_B * T / (np.pi * m))

    # Graficar histograma y ajuste
    plt.figure(figsize=(10, 6))
    plt.hist(speeds, bins=30, density=True, alpha=0.6, color='g', label='Datos simulados')
    plt.plot(bin_centers, maxwell_boltzmann(bin_centers, *popt, m), 'r-', label=f'Ajuste Maxwell-Boltzmann\nT = {popt[0]:.3f}')
    
    # Añadir líneas punteadas para v_p y v_mean
    plt.axvline(v_p, color='b', linestyle='dotted', linewidth=2, label=f'Velocidad más probable = {v_p:.3e}')
    plt.axvline(v_mean, color='m', linestyle='dotted', linewidth=2, label=f'Velocidad media = {v_mean:.3e}')
    
    plt.title('Distribución de la magnitud de las velocidades')
    plt.xlabel('Velocidad')
    plt.ylabel('Densidad')
    plt.legend()
    plt.show()

# Generar posiciones y velocidades iniciales
initial_positions = generate_initial_positions(N)
initial_velocities = generate_velocity_maxwell_boltzmann(T, N)

# Simular el movimiento de las partículas
final_positions = simulate_particles(initial_positions, initial_velocities, dt)

# Graficar posiciones iniciales y finales
plot_positions(initial_positions, final_positions)

# Graficar velocidades y realizar ajuste
plot_velocities_with_fit(initial_velocities)
