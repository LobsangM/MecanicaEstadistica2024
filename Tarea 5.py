import numpy as np
import matplotlib.pyplot as plt

# Constantes físicas
hbar = 1.0545718e-34  # Constante de Planck reducida (J*s)
h = hbar
c = 3e8               # Velocidad de la luz (m/s)
k = 1.380649e-23      # Constante de Boltzmann (J/K)
pi = np.pi

# Función para μ(ω)
def mu_omega(omega, T):
    beta = 1 / (k * T)
    return (hbar * omega**3) / (pi**2 * c**3 * (np.exp(beta * hbar * omega) - 1))

# Rango de ω
omega = np.linspace(1e12, 1e16, 1000)

# Temperaturas
temperatures = [200, 250, 300]

# Graficar μ(ω) para diferentes temperaturas
plt.figure(figsize=(10, 6))

for T in temperatures:
    mu = mu_omega(omega, T)
    plt.plot(omega, mu, label=f'T = {T} K')

plt.xlabel('ω (rad/s)')
plt.ylabel('μ(ω)')
plt.title('Densidad Espectral de Energía Volumétrica μ(ω) para diferentes T')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.show()

# Función para μ(ν)
def mu_nu(nu, T):
    beta = 1 / (k * T)
    return (8 * pi * h * nu**3) / (c**3 * (np.exp(beta * h * nu) - 1))

# Función para μ(λ)
def mu_lambda(lambd, T):
    beta = 1 / (k * T)
    return (8 * pi * h * c) / (lambd**5 * (np.exp(beta * h * c / lambd) - 1))

# Rango de ν y λ
nu = np.linspace(1e12, 1e16, 1000)
lambd = np.linspace(1e-7, 1e-4, 1000)  # de 0.1 micrómetros a 100 micrómetros

# Graficar μ(ν) para diferentes temperaturas
plt.figure(figsize=(10, 6))

for T in temperatures:
    mu = mu_nu(nu, T)
    plt.plot(nu, mu, label=f'T = {T} K')

plt.xlabel('ν (Hz)')
plt.ylabel('μ(ν)')
plt.title('Densidad Espectral de Energía Volumétrica μ(ν) para diferentes T')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.show()

# Graficar μ(λ) para diferentes temperaturas
plt.figure(figsize=(10, 6))

for T in temperatures:
    mu = mu_lambda(lambd, T)
    plt.plot(lambd, mu, label=f'T = {T} K')

plt.xlabel('λ (m)')
plt.ylabel('μ(λ)')
plt.title('Densidad Espectral de Energía Volumétrica μ(λ) para diferentes T')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.show()

