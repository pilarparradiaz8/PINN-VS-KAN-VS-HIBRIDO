#CODIGO PINN (4 CAPAS Y 64 NUERONAS)

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation

# =========================================================
#  0.0 SEMILLA GLOBAL (REPRODUCIBILIDAD TOTAL)
# =========================================================
SEED = 42

import random
import os

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Determinismo en CUDA
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# PyTorch reproducible ops (si está disponible)
try:
    torch.use_deterministic_algorithms(True)
except:
    print("⚠️ Algunas operaciones no son totalmente deterministas")

# Variables de entorno (importante en GPU)
os.environ["PYTHONHASHSEED"] = str(SEED)

# =========================================================
# 0. CONFIGURACIÓN DEL HARDWARE (KAGGLE GPU)
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Dispositivo de cómputo detectado: {device.type.upper()}")

# =========================================================
# 1. RED NEURONAL BASE
# =========================================================
class NLS_Net(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.activation = nn.Tanh()
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])

    def forward(self, x):
        for i in range(len(self.linears)-1):
            x = self.activation(self.linears[i](x))
        return self.linears[-1](x)

# =========================================================
# 2. CEREBRO DOBLE (PINN CONTROL CUÁNTICO)
# =========================================================
class PINN_Control_Cuantico:
    def __init__(self):
        # Movemos las redes a la tarjeta gráfica (GPU)
        capas_psi = [2, 64, 64, 64, 64, 2]
        self.net_psi = NLS_Net(capas_psi).to(device)

        capas_control = [1,  64, 64, 64, 64, 1]
        self.net_control = NLS_Net(capas_control).to(device)

        parametros_totales = list(self.net_psi.parameters()) + list(self.net_control.parameters())
        self.optimizer = torch.optim.Adam(parametros_totales, lr=0.001)

    def net_f(self, x, t):
        # Aseguramos que x y t exigen cálculo de gradiente
        x.requires_grad_(True)
        t.requires_grad_(True)

        psi = self.net_psi(torch.cat([x, t], dim=1))
        psi_R = psi[:, 0:1]
        psi_I = psi[:, 1:2]

        u = self.net_control(t)

        psi_R_t = torch.autograd.grad(psi_R, t, grad_outputs=torch.ones_like(psi_R), create_graph=True)[0]
        psi_I_t = torch.autograd.grad(psi_I, t, grad_outputs=torch.ones_like(psi_I), create_graph=True)[0]

        psi_R_x = torch.autograd.grad(psi_R, x, grad_outputs=torch.ones_like(psi_R), create_graph=True)[0]
        psi_R_xx = torch.autograd.grad(psi_R_x, x, grad_outputs=torch.ones_like(psi_R_x), create_graph=True)[0]

        psi_I_x = torch.autograd.grad(psi_I, x, grad_outputs=torch.ones_like(psi_I), create_graph=True)[0]
        psi_I_xx = torch.autograd.grad(psi_I_x, x, grad_outputs=torch.ones_like(psi_I_x), create_graph=True)[0]

        f_R = psi_I_t + psi_R_xx + u * x * psi_R
        f_I = -psi_R_t + psi_I_xx + u * x * psi_I

        return f_R, f_I

    def compute_loss(self, x_f, t_f, x_b, t_b, x_ini, t_ini, phi_1, x_fin, t_fin, phi_2, x_norm, t_norm, N_t_norm, N_x_norm):
        f_R, f_I = self.net_f(x_f, t_f)
        loss_f = torch.mean(f_R**2 + f_I**2)

        psi_b = self.net_psi(torch.cat([x_b, t_b], dim=1))
        loss_b = torch.mean(psi_b**2)

        psi_ini = self.net_psi(torch.cat([x_ini, t_ini], dim=1))
        loss_ini = torch.mean((psi_ini[:, 0:1] - phi_1)**2 + psi_ini[:, 1:2]**2)

        psi_fin = self.net_psi(torch.cat([x_fin, t_fin], dim=1))
        loss_fin = torch.mean((psi_fin[:, 0:1] - phi_2)**2 + psi_fin[:, 1:2]**2)

        # Conservación de la Norma
        psi_norm = self.net_psi(torch.cat([x_norm, t_norm], dim=1))
        prob_norm = psi_norm[:, 0]**2 + psi_norm[:, 1]**2
        prob_matriz = prob_norm.view(N_t_norm, N_x_norm)
        dx = 1.0 / (N_x_norm - 1)
        integrales_en_el_tiempo = torch.sum(prob_matriz, dim=1) * dx
        loss_norm = torch.mean((integrales_en_el_tiempo - 1.0)**2)

        # Multa máxima a la norma (250.0) para salto rápido
        # PESOS REEQUILIBRADOS:
        # Subimos el inicio y final a 50.0, y bajamos la norma a 100.0
       # Castigo masivo a la pérdida de física:
        return loss_f + (50.0 * loss_b) + (100.0 * loss_ini) + (200.0 * loss_fin) + (5000.0 * loss_norm)

# =========================================================
# 3. GENERACIÓN DE DATOS DIRECTOS EN LA GPU
# =========================================================
print("Generando y subiendo el tejido espacio-temporal a la GPU...")
T_final = 1.0

N_f = 5000
N_b = 1000
N_ini = 1000
N_fin = 1000

# Inyectamos los tensores directamente a la memoria de la GPU
x_f = torch.rand(N_f, 1).to(device).requires_grad_(True)
t_f = (torch.rand(N_f, 1) * T_final).to(device).requires_grad_(True)

t_b_rand = torch.rand(N_b, 1).to(device) * T_final
x_b = torch.cat([torch.zeros(N_b, 1).to(device), torch.ones(N_b, 1).to(device)], dim=0)
t_b = torch.cat([t_b_rand, t_b_rand.clone()], dim=0)

x_ini = torch.rand(N_ini, 1).to(device)
t_ini = torch.zeros(N_ini, 1).to(device)
phi_1 = (math.sqrt(2.0) * torch.sin(math.pi * x_ini)).to(device)

x_fin = torch.rand(N_fin, 1).to(device)
t_fin = (torch.ones(N_fin, 1) * T_final).to(device)
phi_2 = (math.sqrt(2.0) * torch.sin(2.0 * math.pi * x_fin)).to(device)

# --- SOBREMUESTREO ADAPTATIVO EN LA GPU ---
N_t_norm = 400
N_x_norm = 1000

N_t_estable = int(N_t_norm * 0.3)
t_estable = torch.rand(N_t_estable).to(device) * (0.7 * T_final)

N_t_critico = N_t_norm - N_t_estable
t_critico = (torch.rand(N_t_critico).to(device) * (0.3 * T_final)) + (0.7 * T_final)

t_norm_vals = torch.cat([t_estable, t_critico], dim=0)
x_norm_vals = torch.linspace(0, 1.0, N_x_norm).to(device)

T_grid, X_grid = torch.meshgrid(t_norm_vals, x_norm_vals, indexing='ij')
t_norm = T_grid.reshape(-1, 1)
x_norm = X_grid.reshape(-1, 1)

# =========================================================
# 4. ENTRENAMIENTO ACELERADO
# =========================================================
modelo_control = PINN_Control_Cuantico()
epochs = 15000

print(f"Encendiendo el láser... ¡Comienza el entrenamiento masivo en {device.type.upper()}!")
for ep in range(epochs):
    modelo_control.optimizer.zero_grad()
    loss = modelo_control.compute_loss(
        x_f, t_f, x_b, t_b, x_ini, t_ini, phi_1, x_fin, t_fin, phi_2,
        x_norm, t_norm, N_t_norm, N_x_norm)
    loss.backward()
    modelo_control.optimizer.step()

    if ep % 500 == 0:
        print(f"Epoch {ep:5d}, Loss total: {loss.item():.5f}")

# =========================================================
# 5. GRÁFICAS DEL LÁSER Y DEL SALTO
# =========================================================
print("Calculando las predicciones finales para graficar...")
# Creamos puntos en la GPU
t_plot = torch.linspace(0, T_final, 500).view(-1, 1).to(device)
x_plot = torch.linspace(0, 1.0, 200).view(-1, 1).to(device)

with torch.no_grad():
    u_laser = modelo_control.net_control(t_plot)
    dt = T_final / (len(t_plot)-1)
    energia_control = torch.sum(u_laser**2) * dt

    psi_inicio = modelo_control.net_psi(torch.cat([x_plot, torch.zeros_like(x_plot)], dim=1))
    psi_final  = modelo_control.net_psi(torch.cat([x_plot, torch.ones_like(x_plot) * T_final], dim=1))

    prob_inicio = psi_inicio[:, 0]**2 + psi_inicio[:, 1]**2
    prob_final  = psi_final[:, 0]**2 + psi_final[:, 1]**2

fig, axs = plt.subplots(1, 3, figsize=(15, 4))

# Movemos a CPU para Matplotlib
axs[0].plot(t_plot.cpu().numpy(), u_laser.cpu().numpy(), color='purple', linewidth=2)
axs[0].set_title(f"Láser $u(t)$ (Energía: {energia_control.item():.2f})")
axs[0].set_xlabel("Tiempo (t)")
axs[0].grid(True)

prob_1_exacta = (math.sqrt(2.0) * np.sin(math.pi * x_plot.cpu().numpy()))**2
prob_2_exacta = (math.sqrt(2.0) * np.sin(2.0 * math.pi * x_plot.cpu().numpy()))**2

axs[1].plot(x_plot.cpu().numpy(), prob_inicio.cpu().numpy(), color='blue', label="PINN (t=0)")
axs[1].plot(x_plot.cpu().numpy(), prob_1_exacta, '--', color='red', label="Teoría Nivel 1")
axs[1].set_title("Inicio: Estado Fundamental")
axs[1].set_xlabel("Posición (x)")
axs[1].legend(); axs[1].grid(True)

axs[2].plot(x_plot.cpu().numpy(), prob_final.cpu().numpy(), color='green', label=f"PINN (t={T_final})")
axs[2].plot(x_plot.cpu().numpy(), prob_2_exacta, '--', color='red', label="Teoría Nivel 2")
axs[2].set_title(f"Final: Salto Cuántico en {T_final}s")
axs[2].set_xlabel("Posición (x)")
axs[2].legend(); axs[2].grid(True)

for i in [1, 2]:
    ymin, ymax = axs[i].get_ylim()
    axs[i].set_yticks(np.arange(
        np.floor(ymin*2)/2,
        np.ceil(ymax*2)/2 + 0.5,
        0.5
    ))

plt.tight_layout()
plt.show()

# =========================================================
# 6. VERIFICACIÓN DE LA CONSERVACIÓN DE LA NORMA + DOBLE INTEGRAL
# =========================================================

normas_totales = []
doble_integral_acumulada = []

dx_plot = 1.0 / (len(x_plot) - 1)
dt_plot = T_final / (len(t_plot) - 1)

# Convertimos t_plot a CPU para iterar sin problemas de GPU
t_plot_cpu = t_plot.cpu().numpy().flatten()
x_plot_cpu = x_plot.cpu().numpy().flatten()

with torch.no_grad():
    # Recorremos cada tiempo para calcular la norma y la doble integral
    for t_val in t_plot_cpu:
        t_tensor = torch.ones_like(x_plot) * t_val
        psi = modelo_control.net_psi(torch.cat([x_plot, t_tensor], dim=1))

        # Probabilidad en cada posición
        prob = psi[:, 0]**2 + psi[:, 1]**2

        # 1) Norma simple (integral sobre x)
        norma_en_t = torch.sum(prob) * dx_plot
        normas_totales.append(norma_en_t.item())

        # 2) Doble integral: acumulando integral ∫ x*|ψ|² dx * du/dt
        # Aquí centramos la onda
        prob_np = prob.cpu().numpy().flatten()
        centro_gravedad = np.sum(x_plot_cpu * prob_np) * dx_plot
        doble_integral_acumulada.append(centro_gravedad)

# Graficamos la conservación de la norma
plt.figure(figsize=(8, 4))
plt.plot(t_plot_cpu, normas_totales, color='black', linewidth=2.5, label=r'$\int |\psi|^2 dx$ (IA)')
plt.axhline(y=1.0, color='red', linestyle='--', label='Teoría Exacta (1.0)')
plt.ylim(0.9, 1.1)
plt.title("Conservación de la Probabilidad Total")
plt.xlabel("Tiempo (t)")
plt.ylabel("Probabilidad Total")
plt.legend(loc='lower left')
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.show()

# Graficamos el centro de gravedad de la onda (como doble integral acumulada)
plt.figure(figsize=(8, 4))
plt.plot(t_plot_cpu, doble_integral_acumulada, color='orange', linewidth=2.5, label=r'$\int x |\psi|^2 dx$')
plt.title("Centro de Gravedad de la Probabilidad (Doble Integral)")
plt.xlabel("Tiempo (t)")
plt.ylabel(r'$\langle x \rangle$')
plt.legend()
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.show()

# =========================================================
# 7. PROYECCIONES DE FOURIER (HASTA n=5)
# =========================================================
print("Calculando los 5 primeros coeficientes de Fourier...")
def phi_func(n, x):
    return np.sqrt(2) * np.sin(n * np.pi * x)

def coeficiente_cn(modelo, n, x, t):
    with torch.no_grad():
        t_tensor = torch.ones_like(x) * t
        psi = modelo.net_psi(torch.cat([x, t_tensor], 1))
        psi_R = psi[:, 0].cpu().numpy().flatten()
        psi_I = psi[:, 1].cpu().numpy().flatten()

    psi_complex = psi_R + 1j * psi_I
    phi_n = phi_func(n, x.cpu().numpy().flatten())
    dx_f = x[1].item() - x[0].item()
    return np.sum(psi_complex * phi_n) * dx_f

x_fourier = torch.linspace(0, 1, 400).view(-1, 1).to(device)
t_vals = np.linspace(0, T_final, 200)

P1, P2, P3, P4, P5 = [], [], [], [], []

for t in t_vals:
    c1 = coeficiente_cn(modelo_control, 1, x_fourier, t)
    c2 = coeficiente_cn(modelo_control, 2, x_fourier, t)
    c3 = coeficiente_cn(modelo_control, 3, x_fourier, t)
    c4 = coeficiente_cn(modelo_control, 4, x_fourier, t)
    c5 = coeficiente_cn(modelo_control, 5, x_fourier, t)

    P1.append(abs(c1)**2); P2.append(abs(c2)**2); P3.append(abs(c3)**2)
    P4.append(abs(c4)**2); P5.append(abs(c5)**2)

# =========================================================
# 8. GENERACIÓN DE LOS GIFS (CONGELACIÓN AL FINAL)
# =========================================================
print("Preparando los GIFs...")
num_frames_psi = 400
x_numpy_gif = x_plot.cpu().numpy().flatten()
t_grid_gif_psi = torch.linspace(0, T_final, num_frames_psi).to(device)

probabilidades_salto = []
with torch.no_grad():
    for t_val in t_grid_gif_psi:
        t_tensor = torch.ones_like(x_plot) * t_val
        psi = modelo_control.net_psi(torch.cat([x_plot, t_tensor], 1))
        probabilidades_salto.append((psi[:, 0]**2 + psi[:, 1]**2).cpu().numpy().flatten())

fig_psi, ax_psi = plt.subplots(figsize=(10, 6))
fig_psi.suptitle(f"Salto Cuántico: Evolución de $|\psi(x,t)|^2$ (T={T_final}s)", fontsize=16)

phi1_target = (math.sqrt(2.0) * np.sin(math.pi * x_numpy_gif))**2
phi2_target = (math.sqrt(2.0) * np.sin(2.0 * math.pi * x_numpy_gif))**2

ax_psi.plot(x_numpy_gif, phi1_target, '--', color='gray', alpha=0.5, label="Forma Nivel 1")
ax_psi.plot(x_numpy_gif, phi2_target, ':', color='red', alpha=0.8, label="Forma Nivel 2")

linea_salto_psi, = ax_psi.plot([], [], color='#1f77b4', lw=4, label="$|\psi(x,t)|^2$ (IA)")
time_text_psi = ax_psi.text(0.5, 2.3, '', fontsize=12, horizontalalignment='center', fontweight='bold')

ax_psi.set_xlim(0, 1); ax_psi.set_ylim(0, 2.5)
ax_psi.legend(loc="upper right"); ax_psi.grid(True, linestyle='--')

fotogramas_pausa_psi = 80
total_fotogramas_psi = num_frames_psi + fotogramas_pausa_psi

def update_gif_psi(frame):
    idx = min(frame, num_frames_psi - 1)
    linea_salto_psi.set_data(x_numpy_gif, probabilidades_salto[idx])
    time_text_psi.set_text(f"Tiempo t = {t_grid_gif_psi[idx].item():.3f}s / {T_final}s")

    if idx == num_frames_psi - 1:
        linea_salto_psi.set_color('#2ca02c')
    else:
        linea_salto_psi.set_color('#1f77b4')
    return linea_salto_psi, time_text_psi

ani_salto = animation.FuncAnimation(fig_psi, update_gif_psi, frames=total_fotogramas_psi, interval=25, blit=True)
ani_salto.save("salto_onda_completo.gif", writer='pillow', fps=40)
plt.close(fig_psi)
