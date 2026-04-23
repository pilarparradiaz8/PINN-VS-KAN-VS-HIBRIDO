# =========================================================
# HIBRIDO KAN + PINN (Multi-T)
# =========================================================

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# KAN
# =========================================================
class KANLayer(nn.Module):
    def __init__(self, in_dim, out_dim, F=4):
        super().__init__()
        self.F = F
        self.sin = nn.Parameter(torch.randn(out_dim, in_dim, F)*0.02)
        self.cos = nn.Parameter(torch.randn(out_dim, in_dim, F)*0.02)
        self.base = nn.Linear(in_dim, out_dim)

    def forward(self,x):
        x_t = torch.tanh(x)
        out = self.base(x_t)

        for k in range(1, self.F+1):
            out += torch.einsum('oi,bi->bo', self.sin[:,:,k-1], torch.sin(math.pi*k*x_t))
            out += torch.einsum('oi,bi->bo', self.cos[:,:,k-1], torch.cos(math.pi*k*x_t))
        return out


class KAN(nn.Module):
    def __init__(self,layers):
        super().__init__()
        self.layers = nn.ModuleList([KANLayer(layers[i],layers[i+1]) for i in range(len(layers)-1)])

    def forward(self,x):
        for l in self.layers:
            x = l(x)
        return x

# =========================================================
# PINN
# =========================================================
class PINN(nn.Module):
    def __init__(self,layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(layers[i],layers[i+1]) for i in range(len(layers)-1)])
        self.act = nn.Tanh()

    def forward(self,x):
        for i in range(len(self.layers)-1):
            x = self.act(self.layers[i](x))
        return self.layers[-1](x)

# =========================================================
# MODELO HÍBRIDO (ACTUALIZADO CON HARD CONSTRAINTS)
# =========================================================
class Hybrid(nn.Module):
    def __init__(self):
        super().__init__()

        self.kan_psi = KAN([2,32,32,2])
        self.pinn_psi = PINN([2,64,64,2])

        self.kan_u = KAN([1,16,16,1])
        self.pinn_u = PINN([1,64,64,1])

        self.w_kan = nn.Parameter(torch.tensor(1.0))
        self.w_pinn = nn.Parameter(torch.tensor(1.0))

    def psi(self, x, t):
        xt = torch.cat([x,t], 1)
        raw_psi = self.w_kan * self.kan_psi(xt) + self.w_pinn * self.pinn_psi(xt)
       
        # HARD CONSTRAINT ESPACIAL:
        psi_out = x * (1.0 - x) * raw_psi
        return psi_out

    def u(self, t):
        raw_u = self.w_kan * self.kan_u(t) + self.w_pinn * self.pinn_u(t)
        return raw_u

# =========================================================
# LOSS
# =========================================================
def loss_fn(model, x_f,t_f,x_i,t_i,phi_i,x_fi,t_fi,phi_f,x_norm,t_norm,Nt,Nx):

    x_f = x_f.clone().detach().requires_grad_(True)
    t_f = t_f.clone().detach().requires_grad_(True)

    psi = model.psi(x_f,t_f)
    R,I = psi[:,0:1], psi[:,1:2]

    u = model.u(t_f)

    R_t = torch.autograd.grad(R,t_f,torch.ones_like(R),create_graph=True)[0]
    I_t = torch.autograd.grad(I,t_f,torch.ones_like(I),create_graph=True)[0]

    R_x = torch.autograd.grad(R,x_f,torch.ones_like(R),create_graph=True)[0]
    R_xx = torch.autograd.grad(R_x,x_f,torch.ones_like(R_x),create_graph=True)[0]

    I_x = torch.autograd.grad(I,x_f,torch.ones_like(I),create_graph=True)[0]
    I_xx = torch.autograd.grad(I_x,x_f,torch.ones_like(I_x),create_graph=True)[0]

    fR = I_t + R_xx + u*x_f*R
    fI = -R_t + I_xx + u*x_f*I

    loss_f = torch.mean(fR**2 + fI**2)

    psi_i = model.psi(x_i,t_i)
    loss_i = torch.mean((psi_i[:,0:1]-phi_i)**2 + psi_i[:,1:2]**2)

    psi_f = model.psi(x_fi,t_fi)
    loss_fi = torch.mean((psi_f[:,0:1]-phi_f)**2 + psi_f[:,1:2]**2)

    # norma
    psi_n = model.psi(x_norm, t_norm)
    prob = psi_n[:,0:1]**2 + psi_n[:,1:2]**2
    prob = prob.view(Nt, Nx)

    dx = 1.0 / (Nx - 1)
    norm = torch.sum(prob, dim=1) * dx
    loss_norm = torch.mean((norm - 1.0)**2)

    return loss_f + 100*loss_i + 200*loss_fi + 2000*loss_norm

# =========================================================
# FUNCIÓN PRINCIPAL DE SIMULACIÓN
# =========================================================
def ejecutar_simulacion(T_target):
    print(f"\n{'='*50}")
    print(f"INICIANDO SIMULACIÓN PARA T = {T_target}s")
    print(f"{'='*50}\n")

    Nx=200
    Nt=150

    x_f = torch.rand(3000,1).to(device)
    t_f = torch.rand(3000,1).to(device) * T_target

    x_i = torch.rand(1000,1).to(device)
    t_i = torch.zeros_like(x_i)
    phi_i = torch.sqrt(torch.tensor(2.0))*torch.sin(math.pi*x_i)

    x_fi = torch.rand(1000,1).to(device)
    t_fi = torch.ones_like(x_fi) * T_target
    phi_f = torch.sqrt(torch.tensor(2.0))*torch.sin(2*math.pi*x_fi)

    xg = torch.linspace(0,1,Nx).to(device)
    tg = torch.linspace(0,T_target,Nt).to(device)
    Tg,Xg = torch.meshgrid(tg,xg,indexing='ij')

    x_norm = Xg.reshape(-1,1)
    t_norm = Tg.reshape(-1,1)

    # TRAIN
    model = Hybrid().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(4000):
        opt.zero_grad()
        loss = loss_fn(model, x_f, t_f, x_i, t_i, phi_i, x_fi, t_fi, phi_f, x_norm, t_norm, Nt, Nx)
        loss.backward()
        opt.step()

        if ep % 500 == 0:
            print(f"Epoca: {ep:4d} | Loss: {loss.item():.6f}")

    # =========================================================
    # GRÁFICAS DEL LÁSER Y DEL SALTO
    # =========================================================
    print("Calculando las predicciones finales para graficar...")

    t_plot = torch.linspace(0, T_target, 500).view(-1, 1).to(device)
    x_plot = torch.linspace(0, 1.0, 200).view(-1, 1).to(device)

    with torch.no_grad():
        u_laser = model.u(t_plot)
        energia_control = torch.trapz(u_laser.squeeze()**2, t_plot.squeeze())

        psi_inicio = model.psi(x_plot, torch.zeros_like(x_plot))
        psi_final  = model.psi(x_plot, torch.ones_like(x_plot) * T_target)

        prob_inicio = psi_inicio[:, 0]**2 + psi_inicio[:, 1]**2
        prob_final  = psi_final[:, 0]**2 + psi_final[:, 1]**2

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Resultados para T = {T_target}s", fontsize=16)

    axs[0].plot(t_plot.cpu().numpy(), u_laser.cpu().numpy(), color='purple', linewidth=2)
    axs[0].set_title(f"Láser $u(t)$ (Energía: {energia_control.item():.2f})")
    axs[0].set_xlabel("Tiempo (t)")
    axs[0].grid(True)

    prob_1_exacta = (math.sqrt(2.0) * np.sin(math.pi * x_plot.cpu().numpy()))**2
    prob_2_exacta = (math.sqrt(2.0) * np.sin(2.0 * math.pi * x_plot.cpu().numpy()))**2

    axs[1].plot(x_plot.cpu().numpy(), prob_inicio.cpu().numpy(), color='blue', label="Hybrid (t=0)")
    axs[1].plot(x_plot.cpu().numpy(), prob_1_exacta, '--', color='red', label="Teoría Nivel 1")
    axs[1].set_title("Inicio: Estado Fundamental")
    axs[1].set_xlabel("Posición (x)")
    axs[1].legend(); axs[1].grid(True)

    axs[2].plot(x_plot.cpu().numpy(), prob_final.cpu().numpy(), color='green', label=f"Hybrid (t={T_target})")
    axs[2].plot(x_plot.cpu().numpy(), prob_2_exacta, '--', color='red', label="Teoría Nivel 2")
    axs[2].set_title(f"Final: Salto Cuántico en {T_target}s")
    axs[2].set_xlabel("Posición (x)")
    axs[2].legend(); axs[2].grid(True)

    for i in [1, 2]:
        ymin, ymax = axs[i].get_ylim()
        axs[i].set_yticks(np.arange(np.floor(ymin*2)/2, np.ceil(ymax*2)/2 + 0.5, 0.5))

    plt.tight_layout()
    plt.show()

    # =========================================================
    # NORMA + CENTRO DE GRAVEDAD
    # =========================================================
    normas_totales = []
    doble_integral_acumulada = []

    dx_plot = 1.0 / (len(x_plot) - 1)
    t_plot_cpu = t_plot.cpu().numpy().flatten()
    x_plot_cpu = x_plot.cpu().numpy().flatten()

    with torch.no_grad():
        for t_val in t_plot_cpu:
            psi = model.psi(x_plot, torch.ones_like(x_plot)*t_val)
            prob = psi[:, 0]**2 + psi[:, 1]**2

            norma_en_t = torch.sum(prob) * dx_plot
            normas_totales.append(norma_en_t.item())

            prob_np = prob.cpu().numpy().flatten()
            centro = np.sum(x_plot_cpu * prob_np) * dx_plot
            doble_integral_acumulada.append(centro)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
    fig.suptitle(f"Conservación y Dinámica (T={T_target}s)")
   
    ax1.plot(t_plot_cpu, normas_totales, color='black', linewidth=2.5)
    ax1.axhline(y=1.0, color='red', linestyle='--')
    ax1.set_ylim(0.9, 1.1)
    ax1.set_title("Conservación de la Probabilidad Total")
    ax1.set_xlabel("Tiempo (t)")
    ax1.grid(True, linestyle='--')

    ax2.plot(t_plot_cpu, doble_integral_acumulada, color='orange', linewidth=2.5)
    ax2.set_title("Centro de Gravedad de la Probabilidad")
    ax2.set_xlabel("Tiempo (t)")
    ax2.grid(True, linestyle='--')
    plt.tight_layout()
    plt.show()

    # =========================================================
    # FOURIER
    # =========================================================
    print("Calculando los 5 primeros coeficientes de Fourier...")

    def phi_func(n, x):
        return np.sqrt(2) * np.sin(n * np.pi * x)

    def coeficiente_cn(model, n, x, t):
        with torch.no_grad():
            psi = model.psi(x, torch.ones_like(x)*t)
        psi_R = psi[:, 0].cpu().numpy().flatten()
        psi_I = psi[:, 1].cpu().numpy().flatten()
        psi_complex = psi_R + 1j * psi_I
        phi_n = phi_func(n, x.cpu().numpy().flatten())
        dx_f = x[1].item() - x[0].item()
        return np.sum(psi_complex * phi_n) * dx_f

    x_fourier = torch.linspace(0, 1, 400).view(-1, 1).to(device)
    t_vals = np.linspace(0, T_target, 200)

    P = [[] for _ in range(5)]

    for t in t_vals:
        for n in range(5):
            c = coeficiente_cn(model, n+1, x_fourier, t)
            P[n].append(abs(c)**2)

    plt.figure()
    for i in range(5):
        plt.plot(t_vals, P[i], label=f'n={i+1}')

    plt.legend()
    plt.title(f"Coeficientes de Fourier (T={T_target}s)")
    plt.xlabel("Tiempo (t)")
    plt.show()

    # =========================================================
    # GIF FINAL
    # =========================================================
    print(f"Preparando GIF para T={T_target}s...")

    num_frames = 400
    x_numpy = x_plot.cpu().numpy().flatten()
    t_grid = torch.linspace(0, T_target, num_frames).to(device)

    data = []
    with torch.no_grad():
        for t_val in t_grid:
            psi = model.psi(x_plot, torch.ones_like(x_plot)*t_val)
            data.append((psi[:, 0]**2 + psi[:, 1]**2).cpu().numpy().flatten())

    fig, ax = plt.subplots(figsize=(10, 6))

    phi1 = (math.sqrt(2.0) * np.sin(math.pi * x_numpy))**2
    phi2 = (math.sqrt(2.0) * np.sin(2.0 * math.pi * x_numpy))**2

    ax.plot(x_numpy, phi1, '--', color='gray', alpha=0.5)
    ax.plot(x_numpy, phi2, ':', color='red', alpha=0.8)

    line, = ax.plot([], [], lw=4, color='#1f77b4')
    time_text = ax.text(0.5, 2.3, '', ha='center')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 2.5)
    ax.set_title(f"Evolución Cuántica (Target T={T_target}s)")
    ax.grid(True)

    def update(frame):
        idx = min(frame, num_frames - 1)
        line.set_data(x_numpy, data[idx])
        time_text.set_text(f"t = {t_grid[idx].item():.3f}s")

        if idx == num_frames - 1:
            line.set_color('#2ca02c')
        else:
            line.set_color('#1f77b4')

        return line, time_text

    ani = animation.FuncAnimation(fig, update, frames=num_frames+80, interval=25, blit=True)
    gif_name = f"salto_hibrido_T{int(T_target)}.gif"
    ani.save(gif_name, writer='pillow', fps=40)
    plt.close()
   
    print(f"✅ Proceso terminado. GIF guardado como {gif_name}")


# =========================================================
# EJECUCIÓN CONSECUTIVA DE AMBOS TIEMPOS
# =========================================================
if __name__ == "__main__":
    # Primero ejecuta con el tiempo original de 1 segundo
    ejecutar_simulacion(T_target=1.0)
   
    # Luego ejecuta con el tiempo extendido a 5 segundos
    ejecutar_simulacion(T_target=5.0)
