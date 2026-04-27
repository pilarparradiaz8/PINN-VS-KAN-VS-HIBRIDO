
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

# ================================================================
# ========= NUEVAS FIGURAS Y GIFs (ESTILO unified_all_methods) ===
# ================================================================
print("\n" + "="*60)
print("GENERATING EXTENDED FIGURES & GIFs  (unified_all_methods style)")
print("="*60)

# ----------------------------------------------------------------
# UTILITY — running integral  ∫₀ᵗ |u(s)|² ds
# ----------------------------------------------------------------
def _running_integral(t_arr, u2_arr):
    ri = np.zeros(len(t_arr))
    for i in range(1, len(t_arr)):
        ri[i] = ri[i-1] + 0.5*(u2_arr[i-1]+u2_arr[i])*(t_arr[i]-t_arr[i-1])
    return ri

u2      = u_opt**2
ri      = _running_integral(t_mid, u2)
E_total = ri[-1]

_TIME_XLIM = (0, T)
_NORM_YLIM = (0.95, 1.05)
_u_pad     = max(abs(u_opt.min()), abs(u_opt.max())) * 0.12
_u_ylim    = (u_opt.min() - _u_pad, u_opt.max() + _u_pad)
_u2_ymax   = u2.max() * 1.15
_u2_ylim   = (-_u2_ymax * 0.03, _u2_ymax)
_fi_ymax   = ri[-1] * 1.15
_NLEV      = 5
_lev_labels  = [f"n={n}" for n in range(1, _NLEV+1)]
_bar_colors  = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']

# ----------------------------------------------------------------
# fig01 — Initial state (styled)
# ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, p0, color="blue", lw=2.4, label="GRAPE (t=0)")
ax.plot(x, phi1_ex**2, "--", color="black", lw=1.5, label="Exact Level 1")
ax.set_title(r"Initial State $|\psi(x,0)|^2$ — GRAPE", fontsize=13, fontweight='bold')
ax.set_xlabel("x"); ax.set_ylabel(r"$|\psi(x,0)|^2$")
ax.set_xlim(0, 1); ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig(f"{OUT}/fig01_initial_state.png", dpi=150)
plt.show()
print(f"Saved: {OUT}/fig01_initial_state.png")

# ----------------------------------------------------------------
# fig02 — Final state (styled)
# ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, pf, color="red", lw=2.4, label=f"GRAPE (t=1, F={F_final:.5f})")
ax.plot(x, phi2_ex**2, "--", color="black", lw=1.5, label="Exact Level 2")
ax.set_title(r"Final State $|\psi(x,1)|^2$ — GRAPE", fontsize=13, fontweight='bold')
ax.set_xlabel("x"); ax.set_ylabel(r"$|\psi(x,1)|^2$")
ax.set_xlim(0, 1); ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig(f"{OUT}/fig02_final_state.png", dpi=150)
plt.show()
print(f"Saved: {OUT}/fig02_final_state.png")

# ----------------------------------------------------------------
# fig03 — Initial + Final side by side (1×2)
# ----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("GRAPE — Initial and Final States", fontsize=14, fontweight='bold')

axes[0].plot(x, p0, color="blue", lw=2.4, label="GRAPE (t=0)")
axes[0].plot(x, phi1_ex**2, "--", color="black", lw=1.5, label="Exact Level 1")
axes[0].set_title("Initial State (t=0)")
axes[0].set_xlabel("x"); axes[0].set_ylabel(r"$|\psi(x,0)|^2$")
axes[0].set_xlim(0, 1); axes[0].legend(); axes[0].grid(True)

axes[1].plot(x, pf, color="red", lw=2.4, label=f"GRAPE (t=1, F={F_final:.5f})")
axes[1].plot(x, phi2_ex**2, "--", color="black", lw=1.5, label="Exact Level 2")
axes[1].set_title("Final State (t=1)")
axes[1].set_xlabel("x"); axes[1].set_ylabel(r"$|\psi(x,1)|^2$")
axes[1].set_xlim(0, 1); axes[1].legend(); axes[1].grid(True)

plt.tight_layout()
plt.savefig(f"{OUT}/fig03_states_initial_final.png", dpi=150)
plt.show()
print(f"Saved: {OUT}/fig03_states_initial_final.png")

# ----------------------------------------------------------------
# fig04 — Norm conservation (styled, tight ylim)
# ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(times, norms, color="blue", lw=2, label="GRAPE Norm")
ax.axhline(1.0, color="black", ls="--", lw=1.5, label=r"$|\psi|=1$ Reference")
ax.set_ylim(*_NORM_YLIM)
ax.set_xlim(*_TIME_XLIM)
ax.set_title("Norm Conservation — GRAPE", fontsize=13, fontweight='bold')
ax.set_xlabel("Time (t)"); ax.set_ylabel(r"$\Vert\psi(t)\Vert^2$")
ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig(f"{OUT}/fig04_norm.png", dpi=150)
plt.show()
print(f"Saved: {OUT}/fig04_norm.png")

# ----------------------------------------------------------------
# fig05 — Control signal u(t) (styled)
# ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t_mid, u_opt, color="darkgreen", lw=1.5, label="u(t) GRAPE")
ax.set_title(r"Control Signal $u(t)$ — GRAPE", fontsize=13, fontweight='bold')
ax.set_xlabel("Time (t)"); ax.set_ylabel("$u(t)$")
ax.set_xlim(*_TIME_XLIM); ax.set_ylim(*_u_ylim)
ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig(f"{OUT}/fig05_u_signal.png", dpi=150)
plt.show()
print(f"Saved: {OUT}/fig05_u_signal.png")

# ----------------------------------------------------------------
# fig06 — |u(t)|² with fill (styled like fig07 in unified)
# ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t_mid, u2, color="darkgreen", lw=1.8,
        label=f"$|u(t)|^2$  (total $E = {E_total:.2f}$)")
ax.fill_between(t_mid, u2, color="darkgreen", alpha=0.25)
ax.set_title(r"Control Intensity $|u(t)|^2$ — GRAPE", fontsize=13, fontweight='bold')
ax.set_xlabel("Time (t)"); ax.set_ylabel(r"$|u(t)|^2$")
ax.set_xlim(*_TIME_XLIM); ax.set_ylim(*_u2_ylim)
ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig(f"{OUT}/fig06_u2_signal.png", dpi=150)
plt.show()
print(f"Saved: {OUT}/fig06_u2_signal.png")

# ----------------------------------------------------------------
# fig07 — Accumulated control energy f(t) (styled like fig09 in unified)
# ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t_mid, ri, color="darkgreen", lw=2.2,
        label=f"GRAPE  $f(T)={E_total:.2f}$")
ax.fill_between(t_mid, ri, color="darkgreen", alpha=0.18)
ax.set_title(r"Accumulated Control Energy $f(t)=\int_0^t |u(s)|^2\,ds$ — GRAPE",
             fontsize=13, fontweight='bold')
ax.set_xlabel("$t$", fontsize=12); ax.set_ylabel(r"$f(t)=\int_0^t |u|^2\,ds$", fontsize=12)
ax.set_xlim(*_TIME_XLIM); ax.set_ylim(bottom=0)
ax.legend(fontsize=11); ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(f"{OUT}/fig07_f_t_integral.png", dpi=150)
plt.show()
print(f"Saved: {OUT}/fig07_f_t_integral.png")

# ----------------------------------------------------------------
# fig08 — u(t)  |u(t)|²  f(t)  all in one 1×3 panel
# ----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("GRAPE — Control Signal Summary", fontsize=14, fontweight='bold')

axes[0].plot(t_mid, u_opt, color="darkgreen", lw=1.5)
axes[0].set_title(r"$u(t)$")
axes[0].set_xlabel("t"); axes[0].set_ylabel("$u(t)$")
axes[0].set_xlim(*_TIME_XLIM); axes[0].set_ylim(*_u_ylim); axes[0].grid(True)

axes[1].plot(t_mid, u2, color="darkgreen", lw=1.8)
axes[1].fill_between(t_mid, u2, color="darkgreen", alpha=0.25)
axes[1].set_title(r"$|u(t)|^2$")
axes[1].set_xlabel("t"); axes[1].set_ylabel(r"$|u(t)|^2$")
axes[1].set_xlim(*_TIME_XLIM); axes[1].set_ylim(*_u2_ylim); axes[1].grid(True)

axes[2].plot(t_mid, ri, color="darkgreen", lw=2)
axes[2].fill_between(t_mid, ri, color="darkgreen", alpha=0.18)
axes[2].set_title(r"$f(t)=\int_0^t|u|^2\,ds$  ($f(T)={:.2f}$)".format(E_total))
axes[2].set_xlabel("t"); axes[2].set_ylabel(r"$f(t)$")
axes[2].set_xlim(*_TIME_XLIM); axes[2].set_ylim(bottom=0)
axes[2].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(f"{OUT}/fig08_control_summary.png", dpi=150)
plt.show()
print(f"Saved: {OUT}/fig08_control_summary.png")

# ----------------------------------------------------------------
# fig09 — Fourier coefficients |c_n(t)|² at key time slices
# ----------------------------------------------------------------
_t_slices   = np.linspace(0, T, 7)
_idx_slices = np.linspace(0, N_t, len(_t_slices), dtype=int)

fig, axes = plt.subplots(1, len(_t_slices), figsize=(18, 4), sharey=True)
fig.suptitle(r"Fourier probabilities $|c_n|^2$ at selected times — GRAPE",
             fontsize=13, fontweight='bold')

for ax, k, ts in zip(axes, _idx_slices, _t_slices):
    probs = [abs(traj_opt[k][n])**2 for n in range(_NLEV)]
    ax.bar(_lev_labels, probs, color=_bar_colors, edgecolor='black', linewidth=1.2)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"$t={ts:.2f}$", fontsize=11)
    ax.set_xlabel("Level")
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.axhline(0.2, color='gray', ls=':', lw=1, alpha=0.5)

axes[0].set_ylabel(r"$|c_n|^2$")
plt.tight_layout()
plt.savefig(f"{OUT}/fig09_fourier_snapshots.png", dpi=150)
plt.show()
print(f"Saved: {OUT}/fig09_fourier_snapshots.png")

# ================================================================
# ========================== GIFs ================================
# ================================================================
print("\n" + "="*60)
print("GENERATING GIFs  (unified_all_methods style)")
print("="*60)

# ----------------------------------------------------------------
# GIF 1 — Wave evolution + u(t) vline  (1×2)
# ----------------------------------------------------------------
print("Generating gif01_wave_evolution.gif ...")

_Nw          = 120
_fps_w       = 10
_pause_w     = 30
_frame_idx_w = np.linspace(0, N_t, _Nw, dtype=int)
_t_gif_w     = np.linspace(0, T, _Nw)
_frames_w    = list(range(_Nw)) + [_Nw-1]*_pause_w

fig_w, axes_w = plt.subplots(1, 2, figsize=(14, 5))
fig_w.suptitle(r"GRAPE — Wave Evolution $|\psi(x,t)|^2$",
               fontsize=14, fontweight='bold')
fig_w.patch.set_facecolor('white')
for ax in axes_w: ax.set_facecolor('white')

ax_wl, ax_wr = axes_w

# Left: u(t) + vertical time cursor
ax_wl.plot(t_mid, u_opt, color="darkgreen", lw=1.5, alpha=0.85, label="u(t) GRAPE")
_vline_w, = ax_wl.plot([0.0, 0.0], [-1e9, 1e9], color="red", lw=1.5, ls="--",
                        label="Current t")
ax_wl.set_xlim(*_TIME_XLIM); ax_wl.set_ylim(*_u_ylim)
ax_wl.set_title("Control $u(t)$")
ax_wl.set_xlabel("Time (t)"); ax_wl.set_ylabel("Amplitude")
ax_wl.legend(loc="upper right"); ax_wl.grid(True)

# Right: wave
ax_wr.plot(x, phi1_ex**2, ":", color="blue",  alpha=0.4, label="Level 1 (Start)")
ax_wr.plot(x, phi2_ex**2, ":", color="red",   alpha=0.4, label="Level 2 (Target)")
_line_w, = ax_wr.plot([], [], color="green", lw=3, label=r"$|\psi|^2$ GRAPE")
_txt_w   = ax_wr.text(0.05, 0.88, "", transform=ax_wr.transAxes,
                      fontweight='bold', fontsize=10)
ax_wr.set_xlim(0, 1); ax_wr.set_ylim(0, 2.6)
ax_wr.set_title("Wave Evolution")
ax_wr.set_xlabel("x"); ax_wr.set_ylabel(r"$|\psi(x,t)|^2$")
ax_wr.legend(loc="upper right"); ax_wr.grid(True)

plt.tight_layout()

def _upd_wave(frame):
    k  = _frame_idx_w[frame]
    tc = _t_gif_w[frame]
    prob = np.abs(wf(traj_opt[k]))**2
    _line_w.set_data(x, prob)
    fid = abs(psi_f.conj() @ traj_opt[k])**2
    _txt_w.set_text(f"t={tc:.2f}  F={fid:.4f}")
    _vline_w.set_xdata([tc, tc])
    return _line_w, _txt_w, _vline_w

_anim_w = FuncAnimation(fig_w, _upd_wave, frames=_frames_w, interval=100, blit=True)
_path_w = f"{OUT}/gif01_wave_evolution.gif"
_anim_w.save(_path_w, writer=PillowWriter(fps=_fps_w))
plt.close()
print(f"Saved: {_path_w}")

# ----------------------------------------------------------------
# GIF 2 — Norm evolution  (styled like gif_norm_psi in unified)
# ----------------------------------------------------------------
print("Generating gif02_norm_evolution.gif ...")

_Nn       = 160
_fps_n    = 6
_pause_n  = 45
_frames_n = list(range(_Nn)) + [_Nn-1]*_pause_n

fig_n, ax_n = plt.subplots(figsize=(10, 5))
fig_n.patch.set_facecolor('white'); ax_n.set_facecolor('white')

ax_n.plot(times, norms, color="blue", lw=1.5, alpha=0.18)   # faint reference
ax_n.axhline(1.0, color="black", ls="--", lw=1.3, label="Exact = 1")
ax_n.set_xlim(*_TIME_XLIM); ax_n.set_ylim(*_NORM_YLIM)
ax_n.set_title(r"Norm of $\psi$ — GRAPE", fontsize=13, fontweight='bold')
ax_n.set_xlabel("$t$", fontsize=12); ax_n.set_ylabel(r"$\Vert\psi(t)\Vert^2$", fontsize=12)
ax_n.grid(True, linestyle='--', alpha=0.55)

_ln_n,  = ax_n.plot([], [], color="blue", lw=2.2, label="GRAPE Norm")
_dot_n, = ax_n.plot([], [], 'o', color="blue", ms=6, zorder=5)
_txt_n  = ax_n.text(0.985, 0.97, '', transform=ax_n.transAxes,
                    ha='right', va='top', fontsize=11, fontweight='bold',
                    color="blue",
                    bbox=dict(fc='white', ec="blue", boxstyle='round,pad=0.3',
                              alpha=0.92, lw=1.2))
ax_n.legend(loc='upper left', fontsize=10, framealpha=0.9)
plt.tight_layout()

def _upd_norm(frame):
    k = min(int(frame * (len(times)-1) / (_Nn-1)), len(times)-1)
    _ln_n.set_data(times[:k+1], norms[:k+1])
    _dot_n.set_data([times[k]], [norms[k]])
    _txt_n.set_text(f"GRAPE: $t={times[k]:.3f}$  norm={norms[k]:.4f}")
    return _ln_n, _dot_n, _txt_n

_anim_n = FuncAnimation(fig_n, _upd_norm, frames=_frames_n,
                        interval=int(1000/_fps_n), blit=True)
_path_n = f"{OUT}/gif02_norm_evolution.gif"
_anim_n.save(_path_n, writer=PillowWriter(fps=_fps_n))
plt.close()
print(f"Saved: {_path_n}")

# ----------------------------------------------------------------
# GIF 3 — Fourier bar chart  (styled like gif_fourier_bars in unified)
# ----------------------------------------------------------------
print("Generating gif03_fourier_bars.gif ...")

_Nf       = 80
_fps_f    = 6
_pause_f  = 40
_t_four   = np.linspace(0, T, _Nf)
_gi_four  = np.linspace(0, N_t, _Nf, dtype=int)
_frames_f = list(range(_Nf)) + [_Nf-1]*_pause_f

# |c_n(t)|² comes directly from the basis representation in GRAPE
_P_four = np.zeros((_Nf, _NLEV))
for _i, _k in enumerate(_gi_four):
    for _n in range(_NLEV):
        _P_four[_i, _n] = abs(traj_opt[_k][_n])**2

# Normalise rows so bars sum to 1
_P_norm_f = _P_four.copy()
_s = _P_norm_f.sum(axis=1, keepdims=True)
_s[_s < 1e-12] = 1.0
_P_norm_f /= _s

fig_f, ax_f = plt.subplots(figsize=(8, 6))
fig_f.patch.set_facecolor('white'); ax_f.set_facecolor('white')

_bars_f = ax_f.bar(_lev_labels, _P_norm_f[0], color=_bar_colors,
                   edgecolor='black', linewidth=1.2)
ax_f.set_ylim(0, 1.05)
ax_f.set_ylabel(r"$|c_n|^2$ (normalised)", fontsize=12)
ax_f.set_xlabel("Energy level", fontsize=12)
ax_f.set_title("GRAPE — Fourier probabilities  $t = 0.000$", fontsize=13)
ax_f.grid(axis='y', linestyle='--', alpha=0.6)
ax_f.axhline(1.0/5, color='gray', ls=':', lw=1, alpha=0.5)
plt.tight_layout()

def _upd_four(frame):
    i  = min(frame, _Nf-1)
    tc = _t_four[i]
    for bar, h in zip(_bars_f, _P_norm_f[i]):
        bar.set_height(h)
    ax_f.set_title(f"GRAPE — Fourier probabilities  $t = {tc:.3f}$", fontsize=13)
    return _bars_f

_anim_f = FuncAnimation(fig_f, _upd_four, frames=_frames_f,
                        interval=int(1000/_fps_f), blit=False)
_path_f = f"{OUT}/gif03_fourier_bars.gif"
_anim_f.save(_path_f, writer=PillowWriter(fps=_fps_f))
plt.close()
print(f"Saved: {_path_f}")

# ----------------------------------------------------------------
# GIF 4 — f(t) growing curve  (styled like gif_f_t_integral in unified)
# ----------------------------------------------------------------
print("Generating gif04_f_t_integral.gif ...")

_Ni       = 160
_fps_i    = 5
_pause_i  = 50
_frames_i = list(range(_Ni)) + [_Ni-1]*_pause_i

fig_i, ax_i = plt.subplots(figsize=(10, 5))
fig_i.patch.set_facecolor('white'); ax_i.set_facecolor('white')

ax_i.plot(t_mid, ri, color="darkgreen", lw=1.5, alpha=0.18)   # faint reference
ax_i.set_xlim(*_TIME_XLIM); ax_i.set_ylim(0, _fi_ymax)
ax_i.set_title(r"$f(t) = \int_0^t |u(s)|^2\,ds$ — GRAPE",
               fontsize=13, fontweight='bold')
ax_i.set_xlabel("$t$", fontsize=12)
ax_i.set_ylabel(r"$f(t) = \int_0^t |u|^2\,ds$", fontsize=12)
ax_i.grid(True, linestyle='--', alpha=0.6)

_ln_i,  = ax_i.plot([], [], color="darkgreen", lw=2.5,
                    label=f"GRAPE  $f(T)={E_total:.2f}$")
_dot_i, = ax_i.plot([], [], 'o', color="darkgreen", ms=7, zorder=5)
_txt_i  = ax_i.text(0.97, 0.12, '', transform=ax_i.transAxes,
                    ha='right', va='bottom', fontsize=12, fontweight='bold',
                    bbox=dict(fc='white', ec='gray', boxstyle='round,pad=0.35', alpha=0.95))
ax_i.legend(loc='upper left', fontsize=10, framealpha=0.9)
plt.tight_layout()

def _upd_fi(frame):
    k = min(int(frame * (len(t_mid)-1) / (_Ni-1)), len(t_mid)-1)
    _ln_i.set_data(t_mid[:k+1], ri[:k+1])
    _dot_i.set_data([t_mid[k]], [ri[k]])
    _txt_i.set_text(f"$t={t_mid[k]:.3f}$\n$f(t)={ri[k]:.3f}$")
    return _ln_i, _dot_i, _txt_i

_anim_i = FuncAnimation(fig_i, _upd_fi, frames=_frames_i,
                        interval=int(1000/_fps_i), blit=True)
_path_i = f"{OUT}/gif04_f_t_integral.gif"
_anim_i.save(_path_i, writer=PillowWriter(fps=_fps_i))
plt.close()
print(f"Saved: {_path_i}")

# ----------------------------------------------------------------
# GIF 5 — 2×2 live dashboard: u(t)+vline | wave | norm | f(t)
# ----------------------------------------------------------------
print("Generating gif05_dashboard.gif ...")

_Nd          = 120
_fps_d       = 10
_pause_d     = 30
_frame_idx_d = np.linspace(0, N_t, _Nd, dtype=int)
_t_gif_d     = np.linspace(0, T, _Nd)
_frames_d    = list(range(_Nd)) + [_Nd-1]*_pause_d

fig_d, axes_d = plt.subplots(2, 2, figsize=(14, 10))
fig_d.suptitle("GRAPE — Live Dashboard", fontsize=14, fontweight='bold')
fig_d.patch.set_facecolor('white')
for ax in axes_d.flat: ax.set_facecolor('white')

ax_du = axes_d[0, 0]   # u(t) + time cursor
ax_dw = axes_d[0, 1]   # wave |ψ|²
ax_dn = axes_d[1, 0]   # norm
ax_df = axes_d[1, 1]   # f(t)

# u(t) + vline
ax_du.plot(t_mid, u_opt, color="darkgreen", lw=1.5, alpha=0.85, label="u(t)")
_vline_d, = ax_du.plot([0.0, 0.0], [-1e9, 1e9], color="red", lw=1.5, ls="--",
                        label="Current t")
ax_du.set_xlim(*_TIME_XLIM); ax_du.set_ylim(*_u_ylim)
ax_du.set_title("Control $u(t)$")
ax_du.set_xlabel("t"); ax_du.set_ylabel("Amplitude")
ax_du.legend(loc="upper right"); ax_du.grid(True)

# wave
ax_dw.plot(x, phi1_ex**2, ":", color="blue",  alpha=0.4, label="Level 1 (Start)")
ax_dw.plot(x, phi2_ex**2, ":", color="red",   alpha=0.4, label="Level 2 (Target)")
_line_d, = ax_dw.plot([], [], color="green", lw=3, label=r"$|\psi|^2$")
_txt_dw  = ax_dw.text(0.05, 0.88, "", transform=ax_dw.transAxes,
                      fontweight='bold', fontsize=10)
ax_dw.set_xlim(0, 1); ax_dw.set_ylim(0, 2.6)
ax_dw.set_title(r"Wave $|\psi(x,t)|^2$")
ax_dw.set_xlabel("x"); ax_dw.set_ylabel(r"$|\psi|^2$")
ax_dw.legend(loc="upper right"); ax_dw.grid(True)

# norm (faint background)
ax_dn.plot(times, norms, color="blue", lw=1.5, alpha=0.18)
ax_dn.axhline(1.0, color="black", ls="--", lw=1.3)
ax_dn.set_xlim(*_TIME_XLIM); ax_dn.set_ylim(*_NORM_YLIM)
ax_dn.set_title("Norm Conservation")
ax_dn.set_xlabel("t"); ax_dn.set_ylabel(r"$\Vert\psi\Vert^2$")
ax_dn.grid(True, linestyle='--', alpha=0.55)
_ln_dn,  = ax_dn.plot([], [], color="blue", lw=2.2)
_dot_dn, = ax_dn.plot([], [], 'o', color="blue", ms=5, zorder=5)

# f(t) (faint background)
ax_df.plot(t_mid, ri, color="darkgreen", lw=1.5, alpha=0.18)
ax_df.set_xlim(*_TIME_XLIM); ax_df.set_ylim(0, _fi_ymax)
ax_df.set_title(r"Accumulated Energy $f(t)$")
ax_df.set_xlabel("t"); ax_df.set_ylabel(r"$f(t)$")
ax_df.grid(True, linestyle='--', alpha=0.55)
_ln_df,  = ax_df.plot([], [], color="darkgreen", lw=2.2)
_dot_df, = ax_df.plot([], [], 'o', color="darkgreen", ms=5, zorder=5)
_txt_df  = ax_df.text(0.97, 0.12, '', transform=ax_df.transAxes,
                      ha='right', va='bottom', fontsize=11, fontweight='bold',
                      bbox=dict(fc='white', ec='gray', boxstyle='round,pad=0.3', alpha=0.9))

plt.tight_layout()

def _upd_dash(frame):
    k_traj = _frame_idx_d[frame]
    tc     = _t_gif_d[frame]

    # wave + fidelity
    prob = np.abs(wf(traj_opt[k_traj]))**2
    _line_d.set_data(x, prob)
    fid = abs(psi_f.conj() @ traj_opt[k_traj])**2
    _txt_dw.set_text(f"t={tc:.2f}  F={fid:.4f}")
    _vline_d.set_xdata([tc, tc])

    # norm
    k_t = min(int(tc * N_t), N_t)
    _ln_dn.set_data(times[:k_t+1], norms[:k_t+1])
    _dot_dn.set_data([times[k_t]], [norms[k_t]])

    # f(t)
    k_mid = min(int(tc * (len(t_mid)-1) / T), len(t_mid)-1)
    _ln_df.set_data(t_mid[:k_mid+1], ri[:k_mid+1])
    _dot_df.set_data([t_mid[k_mid]], [ri[k_mid]])
    _txt_df.set_text(f"$t={t_mid[k_mid]:.3f}$\n$f={ri[k_mid]:.3f}$")

    return _line_d, _txt_dw, _vline_d, _ln_dn, _dot_dn, _ln_df, _dot_df, _txt_df

_anim_d = FuncAnimation(fig_d, _upd_dash, frames=_frames_d, interval=100, blit=True)
_path_d = f"{OUT}/gif05_dashboard.gif"
_anim_d.save(_path_d, writer=PillowWriter(fps=_fps_d))
plt.close()
print(f"Saved: {_path_d}")

# ================================================================
# SUMMARY
# ================================================================
print("\n" + "="*60)
print(f"All outputs saved to: {OUT}")
print("="*60)
_all_files = [
    # ── original outputs (unchanged) ──
    "states_comparison.png",
    "control_fidelity.png",
    "norm_conservation.png",
    "evolution_white.gif",
    # ── new figures ──
    "fig01_initial_state.png",
    "fig02_final_state.png",
    "fig03_states_initial_final.png",
    "fig04_norm.png",
    "fig05_u_signal.png",
    "fig06_u2_signal.png",
    "fig07_f_t_integral.png",
    "fig08_control_summary.png",
    "fig09_fourier_snapshots.png",
    # ── new GIFs ──
    "gif01_wave_evolution.gif",
    "gif02_norm_evolution.gif",
    "gif03_fourier_bars.gif",
    "gif04_f_t_integral.gif",
    "gif05_dashboard.gif",
]
for f in _all_files:
    print(f"  {OUT}/{f}")

print(f"\nGRAPE Final Fidelity  : {F_final:.7f}")
print(f"GRAPE Control Energy  : {E_total:.3f}")
print("="*60)
