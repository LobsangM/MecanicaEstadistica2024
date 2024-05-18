import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# Graficar velocidades
def plot_velocities(velocities):
    # Calcular las magnitudes de las velocidades
    speeds = np.linalg.norm(velocities, axis=1)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Histograma de las magnitudes de las velocidades
    axs[0, 0].hist(speeds, bins=30, density=True, alpha=0.6, color='g')
    axs[0, 0].set_title('Distribución de la magnitud de las velocidades')
    axs[0, 0].set_xlabel('Velocidad')
    axs[0, 0].set_ylabel('Densidad')

    # Histograma de la componente x de las velocidades
    axs[0, 1].hist(velocities[:, 0], bins=30, density=True, alpha=0.6, color='b')
    axs[0, 1].set_title('Distribución de la componente x de las velocidades')
    axs[0, 1].set_xlabel('Velocidad en x')
    axs[0, 1].set_ylabel('Densidad')

    # Histograma de la componente y de las velocidades
    axs[1, 0].hist(velocities[:, 1], bins=30, density=True, alpha=0.6, color='r')
    axs[1, 0].set_title('Distribución de la componente y de las velocidades')
    axs[1, 0].set_xlabel('Velocidad en y')
    axs[1, 0].set_ylabel('Densidad')

    # Histograma de la componente z de las velocidades
    axs[1, 1].hist(velocities[:, 2], bins=30, density=True, alpha=0.6, color='m')
    axs[1, 1].set_title('Distribución de la componente z de las velocidades')
    axs[1, 1].set_xlabel('Velocidad en z')
    axs[1, 1].set_ylabel('Densidad')

    plt.tight_layout()
    plt.show()

# Generar posiciones y velocidades iniciales
initial_positions = generate_initial_positions(N)
initial_velocities = generate_velocity_maxwell_boltzmann(T, N)

# Simular el movimiento de las partículas
final_positions = simulate_particles(initial_positions, initial_velocities, dt)

# Graficar posiciones iniciales y finales
plot_positions(initial_positions, final_positions)

# Graficar velocidades
plot_velocities(initial_velocities)
