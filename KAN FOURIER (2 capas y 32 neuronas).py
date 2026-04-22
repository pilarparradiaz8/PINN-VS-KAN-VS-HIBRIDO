#CODIGO KAN FOURIER (2 CAPAS Y 32 NEURONAS)

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation
import scipy.signal

# =========================================================
# 0. CONFIGURACIÓN DEL HARDWARE
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Dispositivo de cómputo detectado: {device.type.upper()}")
torch.cuda.empty_cache() # Limpiar memoria previa por si acaso

# =========================================================
# 1. CLASE BASE DE LA RED NEURONAL (FOURIER KAN ESTABILIZADO)
# =========================================================
class Capa_Fourier_KAN(nn.Module):
    def __init__(self, in_dim, out_dim, frecuencias=4):
        super().__init__()
        self.frecuencias = frecuencias

        # MEJORA 1: Reducir la varianza inicial para no empezar con ruido caótico
        self.coef_sin = nn.Parameter(torch.randn(out_dim, in_dim, frecuencias) * 0.02)
        self.coef_cos = nn.Parameter(torch.randn(out_dim, in_dim, frecuencias) * 0.02)

        self.peso_base = nn.Linear(in_dim, out_dim)
        self.act_base = nn.SiLU()

    def forward(self, x):
        salida = self.peso_base(self.act_base(x))

        # MEJORA 2: Acotar estrictamente la entrada al dominio trigonométrico
        # Esto evita que las frecuencias exploten en las capas profundas
        x_bound = torch.tanh(x)

        for k in range(1, self.frecuencias + 1):
            seno = torch.sin(math.pi * k * x_bound)
            coseno = torch.cos(math.pi * k * x_bound)
            salida += torch.einsum('oi,bi->bo', self.coef_sin[:, :, k-1], seno)
            salida += torch.einsum('oi,bi->bo', self.coef_cos[:, :, k-1], coseno)
        return salida

class KAN_Net(nn.Module):
    def __init__(self, layers, frecuencias=4):
        super().__init__()
        self.capas = nn.ModuleList([
            Capa_Fourier_KAN(layers[i], layers[i+1], frecuencias)
            for i in range(len(layers)-1)
        ])

    def forward(self, x):
        for capa in self.capas:
            x = capa(x)
        return x

# =========================================================
# 2. CEREBRO DOBLE KAN PARA CONTROL CUÁNTICO
# =========================================================
class KAN_Control_Cuantico:
    def __init__(self):
        capas_psi = [2, 32, 32, 2]
        self.net_psi = KAN_Net(capas_psi, frecuencias=4).to(device)

        capas_control = [1, 16, 16, 1]
        self.net_control = KAN_Net(capas_control, frecuencias=4).to(device)

        parametros_totales = list(self.net_psi.parameters()) + list(self.net_control.parameters())
        # MEJORA 3: LR ligeramente más bajo para la convergencia suave de la KAN
        self.optimizer = torch.optim.Adam(parametros_totales, lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2500, gamma=0.5)

    def get_psi(self, x, t):
        entradas = torch.cat([x, t], dim=1)
        psi_raw = self.net_psi(entradas)
        # Ansatz matemático: Garantiza psi=0 en los bordes espaciales x=0 y x=1
        return psi_raw * (x * (1.0 - x))

    def net_f(self, x, t):
        x.requires_grad_(True)
        t.requires_grad_(True)

        psi = self.get_psi(x, t)
        psi_R = psi[:, 0:1]
        psi_I = psi[:, 1:2]

        u_control = self.net_control(t)

        psi_R_t = torch.autograd.grad(psi_R, t, grad_outputs=torch.ones_like(psi_R), create_graph=True)[0]
        psi_I_t = torch.autograd.grad(psi_I, t, grad_outputs=torch.ones_like(psi_I), create_graph=True)[0]

        psi_R_x = torch.autograd.grad(psi_R, x, grad_outputs=torch.ones_like(psi_R), create_graph=True)[0]
        psi_R_xx = torch.autograd.grad(psi_R_x, x, grad_outputs=torch.ones_like(psi_R_x), create_graph=True)[0]

        psi_I_x = torch.autograd.grad(psi_I, x, grad_outputs=torch.ones_like(psi_I), create_graph=True)[0]
        psi_I_xx = torch.autograd.grad(psi_I_x, x, grad_outputs=torch.ones_like(psi_I_x), create_graph=True)[0]

        f_R = psi_I_t + psi_R_xx + (u_control * x * psi_R)
        f_I = -psi_R_t + psi_I_xx + (u_control * x * psi_I)

        return f_R, f_I

    def compute_loss(self, x_f, t_f, x_ini, t_ini, phi_1, x_fin, t_fin, phi_2, x_norm, t_norm, N_t_norm, N_x_norm):
        f_R, f_I = self.net_f(x_f, t_f)
        loss_f = torch.mean(f_R**2 + f_I**2)

        psi_ini = self.get_psi(x_ini, t_ini)
        loss_ini = torch.mean((psi_ini[:, 0:1] - phi_1)**2 + psi_ini[:, 1:2]**2)

        psi_fin = self.get_psi(x_fin, t_fin)
        loss_fin = torch.mean((psi_fin[:, 0:1] - phi_2)**2 + psi_fin[:, 1:2]**2)

        psi_norm = self.get_psi(x_norm, t_norm)
        prob_norm = psi_norm[:, 0]**2 + psi_norm[:, 1]**2
        prob_matriz = prob_norm.view(N_t_norm, N_x_norm)
        dx = 1.0 / (N_x_norm - 1)
        integrales_en_el_tiempo = torch.sum(prob_matriz, dim=1) * dx
        loss_norm = torch.mean((integrales_en_el_tiempo - 1.0)**2)

        # MEJORA 4: Devolver autoridad a las fronteras. Las KAN necesitan mano dura aquí.
        return loss_f + (250.0 * loss_ini) + (200.0 * loss_fin) + (2500.0 * loss_norm)

# =========================================================
# 3. GENERACIÓN DE DATOS
# =========================================================
print("Generando el tejido espacio-temporal en la GPU...")
T_final = 1.0
N_f = 6000
N_ini = 1000
N_fin = 1000

x_f = torch.rand(N_f, 1).to(device).requires_grad_(True)
t_f = (torch.rand(N_f, 1) * T_final).to(device).requires_grad_(True)

x_ini = torch.rand(N_ini, 1).to(device)
t_ini = torch.zeros(N_ini, 1).to(device)
phi_1 = (math.sqrt(2.0) * torch.sin(math.pi * x_ini)).to(device)

x_fin = torch.rand(N_fin, 1).to(device)
t_fin = (torch.ones(N_fin, 1) * T_final).to(device)
phi_2 = (math.sqrt(2.0) * torch.sin(2.0 * math.pi * x_fin)).to(device)

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
modelo_control = KAN_Control_Cuantico()
epochs = 10000

print(f"Encendiendo el láser KAN estabilizado... ¡Comienza el entrenamiento!")
for ep in range(epochs):
    modelo_control.optimizer.zero_grad()
    loss = modelo_control.compute_loss(
        x_f, t_f, x_ini, t_ini, phi_1, x_fin, t_fin, phi_2,
        x_norm, t_norm, N_t_norm, N_x_norm)
    loss.backward()
    modelo_control.optimizer.step()
    modelo_control.scheduler.step()

    if ep % 500 == 0:
        print(f"Epoch {ep:5d}, Loss total: {loss.item():.5f}, LR: {modelo_control.scheduler.get_last_lr()[0]:.5f}")

# =========================================================
# 5. GRÁFICAS DEL LÁSER Y DEL SALTO
# =========================================================
print("Calculando las predicciones finales para graficar...")

t_plot = torch.linspace(0, T_final, 500).view(-1, 1).to(device)
x_plot = torch.linspace(0, 1.0, 200).view(-1, 1).to(device)

with torch.no_grad():
    u_laser = modelo_control.net_control(t_plot)
    dt = T_final / (len(t_plot)-1)
    energia_control = torch.sum(u_laser**2) * dt

    t_cero = torch.zeros_like(x_plot)
    t_fin_tensor = torch.ones_like(x_plot) * T_final

    psi_inicio = modelo_control.get_psi(x_plot, t_cero)
    psi_final  = modelo_control.get_psi(x_plot, t_fin_tensor)

    prob_inicio = psi_inicio[:, 0]**2 + psi_inicio[:, 1]**2
    prob_final  = psi_final[:, 0]**2 + psi_final[:, 1]**2

x_cpu = x_plot.cpu().numpy()
t_cpu = t_plot.cpu().numpy()

prob_1_exacta = (math.sqrt(2.0) * np.sin(math.pi * x_cpu))**2
prob_2_exacta = (math.sqrt(2.0) * np.sin(2.0 * math.pi * x_cpu))**2

fig, axs = plt.subplots(1, 3, figsize=(15, 4))
axs[0].plot(t_cpu, u_laser.cpu().numpy(), color='purple', linewidth=2)
axs[0].set_title(f"Láser $u(t)$ KAN (Energía: {energia_control.item():.2f})")
axs[0].set_xlabel("Tiempo (t)")
axs[0].grid(True)

axs[1].plot(x_cpu, prob_inicio.cpu().numpy(), color='blue', label="KAN (t=0)")
axs[1].plot(x_cpu, prob_1_exacta, '--', color='red', label="Teoría Nivel 1")
axs[1].set_title("Inicio: Estado Fundamental")
axs[1].set_xlabel("Posición (x)")
axs[1].legend(); axs[1].grid(True)

axs[2].plot(x_cpu, prob_final.cpu().numpy(), color='green', label=f"KAN (t={T_final})")
axs[2].plot(x_cpu, prob_2_exacta, '--', color='red', label="Teoría Nivel 2")
axs[2].set_title(f"Final: Salto Cuántico en {T_final}s")
axs[2].set_xlabel("Posición (x)")
axs[2].legend(); axs[2].grid(True)
plt.tight_layout()
plt.show()

# =========================================================
# 6. VERIFICACIÓN DE LA CONSERVACIÓN DE LA NORMA + CENTRO DE GRAVEDAD
# =========================================================
normas_totales = []
doble_integral_acumulada = []

dx_plot = 1.0 / (len(x_plot) - 1)
t_plot_cpu = t_plot.cpu().numpy().flatten()
x_plot_cpu = x_plot.cpu().numpy().flatten()

with torch.no_grad():
    for t_val in t_plot_cpu:
        t_tensor = torch.ones_like(x_plot) * t_val
        psi = modelo_control.get_psi(x_plot, t_tensor)
        prob = psi[:, 0]**2 + psi[:, 1]**2

        norma_en_t = torch.sum(prob) * dx_plot
        normas_totales.append(norma_en_t.item())

        prob_np = prob.cpu().numpy().flatten()
        centro_gravedad = np.sum(x_plot_cpu * prob_np) * dx_plot
        doble_integral_acumulada.append(centro_gravedad)

plt.figure(figsize=(8, 4))
plt.plot(t_plot_cpu, normas_totales, color='black', linewidth=2.5, label=r'$\int |\psi|^2 dx$ (KAN)')
plt.axhline(y=1.0, color='red', linestyle='--', label='Teoría Exacta (1.0)')
plt.ylim(0.9, 1.1)
plt.title("Estabilidad KAN: Conservación de la Probabilidad Total")
plt.xlabel("Tiempo (t)"); plt.ylabel("Probabilidad Total")
plt.legend(loc='lower left'); plt.grid(True, linestyle='--')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(t_plot_cpu, doble_integral_acumulada, color='orange', linewidth=2.5, label=r'$\int x |\psi|^2 dx$')
plt.title("Centro de Gravedad de la Probabilidad")
plt.xlabel("Tiempo (t)"); plt.ylabel(r'$\langle x \rangle$')
plt.legend(); plt.grid(True, linestyle='--')
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
        psi = modelo.get_psi(x, t_tensor)
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
fig_bar, ax_bar = plt.subplots(figsize=(8, 6))

niveles = ['n=1', 'n=2', 'n=3', 'n=4', 'n=5']
colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

probs_frame_0 = [P1[0], P2[0], P3[0], P4[0], P5[0]]
barras = ax_bar.bar(niveles, probs_frame_0, color=colores, edgecolor='black', linewidth=1.5)

ax_bar.set_ylim(0, 1.05)
ax_bar.set_ylabel("Probabilidad $|c_n|^2$", fontsize=12)
ax_bar.set_title("Transvase de Probabilidad Cuántica (t = 0.000s)", fontsize=14)
ax_bar.grid(axis='y', linestyle='--', alpha=0.7)

fotogramas_pausa = 40
total_frames_barras = len(t_vals) + fotogramas_pausa

def update_bar(frame):
    idx = min(frame, len(t_vals) - 1)
    probs_actuales = [P1[idx], P2[idx], P3[idx], P4[idx], P5[idx]]
    for barra, prob in zip(barras, probs_actuales):
        barra.set_height(prob)
    ax_bar.set_title(f"Transvase Cuántico KAN (t = {t_vals[idx]:.3f}s / {T_final}s)", fontsize=14)
    return barras

ani_barras = animation.FuncAnimation(fig_bar, update_bar, frames=total_frames_barras, interval=50, blit=False)
nombre_gif_barras = "diagrama_barras_fourier_KAN.gif"
ani_barras.save(nombre_gif_barras, writer='pillow', fps=20)
plt.close(fig_bar)

num_frames_psi = 400
x_numpy_gif = x_plot.cpu().numpy().flatten()
t_grid_gif_psi = torch.linspace(0, T_final, num_frames_psi).to(device)

probabilidades_salto = []
with torch.no_grad():
    for t_val in t_grid_gif_psi:
        t_tensor = torch.ones_like(x_plot) * t_val
        psi = modelo_control.get_psi(x_plot, t_tensor)
        probabilidades_salto.append((psi[:, 0]**2 + psi[:, 1]**2).cpu().numpy().flatten())

fig_psi, ax_psi = plt.subplots(figsize=(10, 6))
fig_psi.suptitle(f"Salto Cuántico KAN: Evolución de $|\psi(x,t)|^2$ (T={T_final}s)", fontsize=16)

phi1_target = (math.sqrt(2.0) * np.sin(math.pi * x_numpy_gif))**2
phi2_target = (math.sqrt(2.0) * np.sin(2.0 * math.pi * x_numpy_gif))**2

ax_psi.plot(x_numpy_gif, phi1_target, '--', color='gray', alpha=0.5, label="Forma Nivel 1")
ax_psi.plot(x_numpy_gif, phi2_target, ':', color='red', alpha=0.8, label="Forma Nivel 2")

linea_salto_psi, = ax_psi.plot([], [], color='#1f77b4', lw=4, label="$|\psi(x,t)|^2$ (KAN)")
time_text_psi = ax_psi.text(0.5, 2.3, '', fontsize=12, horizontalalignment='center', fontweight='bold')

ax_psi.set_xlim(0, 1); ax_psi.set_ylim(0, 2.5)
ax_psi.legend(loc="upper right"); ax_psi.grid(True, linestyle='--')

total_fotogramas_psi = num_frames_psi + fotogramas_pausa

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
ani_salto.save("salto_onda_completo_KAN.gif", writer='pillow', fps=40)
plt.close(fig_psi)
print("✅ ¡GIFs guardados con éxito!")

# =========================================================
# 9. ANÁLISIS TERMODINÁMICO: TRABAJO VS ENERGÍA INTERNA
# =========================================================
print("\nVerificando la Termodinámica (Trabajo Integral Trapecio vs Energía Fourier)...")

E_niveles = np.array([1**2, 2**2, 3**2, 4**2, 5**2]) * math.pi**2
E1_teorica = E_niveles[0]
E2_teorica = E_niveles[1]
salto_teorico = E2_teorica - E1_teorica

dt_calc = t_vals[1] - t_vals[0]
dx_calc = 1.0 / (len(x_plot) - 1)
x_np = x_plot.cpu().numpy().flatten()

t_tensor_laser = torch.tensor(t_vals, dtype=torch.float32).view(-1, 1).to(device)
with torch.no_grad():
    u_laser_tvals = modelo_control.net_control(t_tensor_laser).cpu().numpy().flatten()

u_laser_tvals = scipy.signal.savgol_filter(u_laser_tvals, 51, 3)

trabajo_acumulado = []
trabajo_total = 0.0
centro_x_prev = None

aumento_energia_interna = []

with torch.no_grad():
    for i, t_val in enumerate(t_vals):

        t_tensor = torch.ones_like(x_plot) * t_val
        psi = modelo_control.get_psi(x_plot, t_tensor)
        prob_np = (psi[:, 0]**2 + psi[:, 1]**2).cpu().numpy().flatten()

        centro_x = np.sum(x_np * prob_np) * dx_calc

        if i > 0:
            du = u_laser_tvals[i] - u_laser_tvals[i-1]
            trabajo_total += 0.5 * (centro_x + centro_x_prev) * du

        trabajo_acumulado.append(trabajo_total)
        centro_x_prev = centro_x

        E_t = (P1[i]*E_niveles[0] +
               P2[i]*E_niveles[1] +
               P3[i]*E_niveles[2] +
               P4[i]*E_niveles[3] +
               P5[i]*E_niveles[4])

        aumento_energia_interna.append(E_t - E1_teorica)

print(f"-> Salto Teórico esperado: {salto_teorico:.4f} J")
print(f"-> Aumento de Energía (Fourier): {aumento_energia_interna[-1]:.4f} J")
print(f"-> Trabajo Total inyectado (Integral): {trabajo_total:.4f} J")

plt.figure(figsize=(10, 6))
plt.plot(t_vals, trabajo_acumulado, color='cyan', lw=5, alpha=0.7, label='Trabajo del Láser ($W$)')
plt.plot(t_vals, aumento_energia_interna, color='magenta', lw=2, linestyle='--', label='Aumento Energía de la Onda ($\Delta E$)')

plt.axhline(y=salto_teorico, color='red', linestyle=':', label=f'Meta Teórica ($3\pi^2 \\approx 29.61$)')

plt.title("Prueba Termodinámica KAN: Primera Ley ($W = \Delta E$)")
plt.xlabel("Tiempo (t)")
plt.ylabel("Energía Transmitida (J)")
plt.legend(loc="upper left")
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.show()
