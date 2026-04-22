"""
# ===============================================================
# GRAPE QUANTUM CONTROL — PYTORCH ACCELERATED & WHITE STYLE
# Compatible with Kaggle / Google Colab / local Python
# ===============================================================
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.optimize import minimize
from IPython.display import display, Image
import os
import time

# ===============================================================
# OUTPUT DIRECTORY
# ===============================================================
if os.path.exists('/kaggle/working'):
    OUT = '/kaggle/working'
elif os.path.exists('/content'):
    OUT = '/content'
else:
    OUT = './outputs'

os.makedirs(OUT, exist_ok=True)

# ===============================================================
# PYTORCH DEVICE SETUP
# ===============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Hardware computing using: {device.type.upper()}")

# ===============================================================
# PARAMETERS & SYSTEM
# ===============================================================
T = 1.0
N_t = 500
dt = T / N_t
N_basis = 15
n_arr = np.arange(1, N_basis + 1)
E_arr = (n_arr * np.pi) ** 2
H0 = np.diag(E_arr)

def x_mel(m, n):
    if m == n: return 0.5
    pm, pp = m - n, m + n
    return ((-1)**pm - 1)/(pm*np.pi)**2 - ((-1)**pp - 1)/(pp*np.pi)**2

V_mat = np.array([[x_mel(m+1, n+1) for n in range(N_basis)] for m in range(N_basis)])
psi0 = np.zeros(N_basis, dtype=complex); psi0[0] = 1.0
psi_f = np.zeros(N_basis, dtype=complex); psi_f[1] = 1.0

# Pre-cargamos las matrices base en la GPU con precisión doble (complex128)
H0_t = torch.tensor(H0, dtype=torch.complex128, device=device)
V_mat_t = torch.tensor(V_mat, dtype=torch.complex128, device=device)
psi0_t = torch.tensor(psi0, dtype=torch.complex128, device=device)
psi_f_t = torch.tensor(psi_f, dtype=torch.complex128, device=device)

# ===============================================================
# FUNCTIONS (PYTORCH ACCELERATED)
# ===============================================================
def forward(u):
    """Calcula la trayectoria usando PyTorch y la devuelve como lista de arrays Numpy"""
    u_t = torch.tensor(u, dtype=torch.complex128, device=device)

    # Construcción y diagonalización en lote (batch)
    H_batch = H0_t.unsqueeze(0) + u_t.view(-1, 1, 1) * V_mat_t.unsqueeze(0)
    ev, ec = torch.linalg.eigh(H_batch)
    U_batch = ec @ (torch.exp(-1j * ev * dt).unsqueeze(-1) * ec.mH)

    psi = psi0_t
    traj = [psi.cpu().numpy()]
    for k in range(N_t):
        psi = U_batch[k] @ psi
        traj.append(psi.cpu().numpy())
    return traj

def neg_F(u_flat):
    """Calcula la función de pérdida y el gradiente usando operaciones GPU vectorizadas"""
    u_t = torch.tensor(u_flat, dtype=torch.complex128, device=device)

    # 1. Propagadores en lote
    H_batch = H0_t.unsqueeze(0) + u_t.view(-1, 1, 1) * V_mat_t.unsqueeze(0)
    ev, ec = torch.linalg.eigh(H_batch)
    U_batch = ec @ (torch.exp(-1j * ev * dt).unsqueeze(-1) * ec.mH)

    # 2. Paso hacia adelante (Forward) todo en tensores
    traj = torch.zeros((N_t + 1, N_basis), dtype=torch.complex128, device=device)
    traj[0] = psi0_t
    psi = psi0_t
    for k in range(N_t):
        psi = U_batch[k] @ psi
        traj[k+1] = psi

    # 3. Cálculo de la Fidelidad
    lam = torch.vdot(psi_f_t, traj[-1])
    F = torch.abs(lam)**2

    # 4. Paso hacia atrás (Backward) para el Gradiente Analítico
    chi = lam * psi_f_t
    grad = torch.zeros(N_t, dtype=torch.float64, device=device)

    for k in range(N_t-1, -1, -1):
        grad[k] = 2 * dt * torch.imag(torch.vdot(chi, V_mat_t @ traj[k]))
        chi = U_batch[k].mH @ chi

    # Extraemos el valor a NumPy para que SciPy Optimize lo entienda
    return -F.item(), -grad.cpu().numpy()

# ===============================================================
# OPTIMIZATION
# ===============================================================
omega_res = E_arr[1] - E_arr[0]
t_mid = (np.arange(N_t) + 0.5) * dt
env = np.exp(-0.5 * ((t_mid - 0.5)/(0.22))**2)
np.random.seed(7)
u0 = 25.0 * env * np.sin(omega_res * t_mid) + 0.8 * np.random.randn(N_t)

call_log = []
def neg_F_log(u):
    val, grad = neg_F(u)
    call_log.append(-val)
    return val, grad

print("Optimising...")
t_start = time.time()
res = minimize(neg_F_log, u0, method="L-BFGS-B", jac=True, options={"maxiter": 400})
print(f"Optimisation finished in {time.time() - t_start:.2f} seconds!")

u_opt = res.x
F_final = -res.fun
traj_opt = forward(u_opt)

# ===============================================================
# PREPARE PLOTS (STYLE: WHITE)
# ===============================================================
plt.style.use("default") # Fuerza fondo blanco y texto negro
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "black",
    "grid.color": "#DDDDDD"
})

N_x = 700
x = np.linspace(0, 1, N_x)
Phi = np.sqrt(2) * np.sin(np.outer(n_arr, np.pi * x))
def wf(c): return c @ Phi
phi1_ex = np.sqrt(2) * np.sin(np.pi*x)
phi2_ex = np.sqrt(2) * np.sin(2*np.pi*x)
times = np.linspace(0, 1, N_t+1)
norms = np.array([np.real(v.conj() @ v) for v in traj_opt])

# --- FIG 1: COMPARISON ---
fig, ax = plt.subplots(1,2, figsize=(14,5))
p0 = np.abs(wf(traj_opt[0]))**2
ax[0].plot(x, p0, color="blue", lw=2.4, label="GRAPE (t=0)")
ax[0].plot(x, phi1_ex**2, "--", color="black", lw=1.5, label="Exact Target")
ax[0].set_title("Initial State")
ax[0].legend()

pf = np.abs(wf(traj_opt[-1]))**2
ax[1].plot(x, pf, color="red", lw=2.4, label="GRAPE (t=1)")
ax[1].plot(x, phi2_ex**2, "--", color="black", lw=1.5, label="Exact Target")
ax[1].set_title(f"Final State (F={F_final:.5f})")
ax[1].legend()

for a in ax: a.set_xlim(0,1); a.grid(True)
plt.savefig(f"{OUT}/states_comparison.png")
plt.show()

# --- FIG 2: NORM & CONTROL ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
ax1.plot(t_mid, u_opt, color="darkgreen", lw=1.5)
ax1.set_title("Optimised Control Signal u(t)")
ax1.set_ylabel("Amplitude")
ax1.grid(True)

ax2.plot(call_log, color="darkblue", lw=2)
ax2.set_title("Fidelity Convergence")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Fidelity")
ax2.grid(True)

plt.tight_layout()
plt.savefig(f"{OUT}/control_fidelity.png")
plt.show()

# --- FIG 3: NORM CONSERVATION ---
plt.figure(figsize=(10, 4))
plt.plot(times, norms, color="purple", lw=2)
plt.axhline(1.0, color="black", ls="--", alpha=0.5)
plt.ylim(0.9, 1.1)
plt.title("Norm Conservation (Expected = 1.0)")
plt.grid(True)
plt.savefig(f"{OUT}/norm_conservation.png")
plt.show()

# ===============================================================
# GIF (WHITE BACKGROUND & FROZEN FINAL FRAME)
# ===============================================================
fps = 15
N_frames = 120
frame_idx = np.linspace(0, N_t, N_frames, dtype=int)

# --- Truco para congelar el final ---
# Repetimos el último frame durante 3 segundos (3 seg * 15 fps = 45 frames extra)
pause_frames = 3 * fps
frames_list = list(range(N_frames)) + [N_frames - 1] * pause_frames

fig_g = plt.figure(figsize=(8, 5))
ax_g = fig_g.add_subplot(111)
ax_g.set_facecolor("white")
ax_g.set_xlim(0, 1)
ax_g.set_ylim(0, 2.6)
ax_g.grid(True, color="#EEEEEE")

# Colores contrastados para el GIF
ax_g.plot(x, phi1_ex**2, ":", color="blue", alpha=0.4, label="State 1 (Start)")
ax_g.plot(x, phi2_ex**2, ":", color="red", alpha=0.4, label="State 2 (Target)")
line, = ax_g.plot([], [], color="green", lw=3, label="Evolution")
txt = ax_g.text(0.05, 0.9, "", transform=ax_g.transAxes, fontweight='bold')
ax_g.legend(loc="upper right")

def update(frame):
    # 'frame' toma los valores de la lista 'frames_list', incluyendo los repetidos
    k = frame_idx[frame]
    prob = np.abs(wf(traj_opt[k]))**2
    line.set_data(x, prob)
    txt.set_text(f"Time: {k/N_t:.2f} | Fidelity: {abs(psi_f.conj() @ traj_opt[k])**2:.4f}")
    return line, txt

print("Generando GIF...")
anim = FuncAnimation(fig_g, update, frames=frames_list, interval=80, blit=True)
gif_path = f"{OUT}/evolution_white.gif"
anim.save(gif_path, writer=PillowWriter(fps=fps))
plt.close()

display(Image(filename=gif_path))
print(f"\nProceso finalizado. Archivos guardados en: {OUT}")
