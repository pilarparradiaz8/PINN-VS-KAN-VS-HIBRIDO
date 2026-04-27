
"""
UNIFIED SCRIPT — GRAPE + MLP + KAN + Hybrid MLP-KAN
Four-method Quantum Control Comparison.
Can be executed for different final time T. It can be changed in line 105
Compatible with Kaggle / Google Colab / local Python.

Outputs
-------
fig01_initial_states_2x2.png
fig02_final_states_2x2.png
fig03_states_all_methods.png
fig04_norm_all_in_one.png
fig05_norm_subplots_2x2.png
fig06_u_subplots_2x2.png
fig07_u2_subplots_2x2.png
fig08_u2_all_in_one.png
gif_norm_psi_4methods.gif
gif_wave_evolution_4methods.gif
gif_u2_integral_4panels.gif       ← NEW (4 subplots, slow)
gif_u2_integral_single.gif        ← NEW (single figure, slow)
"""

# ════════════════════════════════════════════════════════════════
#  IMPORTS
# ════════════════════════════════════════════════════════════════
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.optimize import minimize
import math, os, time, random

try:
    from IPython.display import display, Image
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False

# ════════════════════════════════════════════════════════════════
#  OUTPUT DIR + STYLE + DEVICE
# ════════════════════════════════════════════════════════════════
if   os.path.exists('/kaggle/working'): OUT = '/kaggle/working'
elif os.path.exists('/content'):        OUT = '/content'
else:                                   OUT = './outputs'
os.makedirs(OUT, exist_ok=True)
print(f"Output directory: {OUT}")

plt.style.use("default")
plt.rcParams.update({
    "figure.facecolor":"white","axes.facecolor":"white",
    "savefig.facecolor":"white","axes.edgecolor":"black",
    "grid.color":"#DDDDDD"
})

SEED   = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Computing device: {device.type.upper()}")

# ════════════════════════════════════════════════════════════════
#  UTILITY — running integral and u² GIF helper
# ════════════════════════════════════════════════════════════════
def _running_integral(t_arr, u2_arr):
    """Cumulative trapezoid ∫₀ᵗ |u(s)|² ds"""
    ri = np.zeros(len(t_arr))
    for i in range(1, len(t_arr)):
        ri[i] = ri[i-1] + 0.5*(u2_arr[i-1]+u2_arr[i])*(t_arr[i]-t_arr[i-1])
    return ri


def _configure_problem_for_T(T, Nt_grape=500):
    """
    Adapt the temporal horizon from [0,1] to [0,T] while keeping the
    original spatial problem and the same relative temporal sampling.
    """
    T = float(T)
    if T <= 0:
        raise ValueError("T must be strictly positive.")
    dt = T / Nt_grape
    tmid = (np.arange(Nt_grape) + 0.5) * dt
    return {
        "T": T,
        "Nt_grape": Nt_grape,
        "dt_grape": dt,
        "tmid_grape": tmid,
        "grape_env_center": 0.5 * T,
        "grape_env_sigma": 0.22 * T,
    }


def _adaptive_norm_times(T, Ntn, device):
    """
    Same adaptive distribution already used in the original script,
    scaled from [0,1] to [0,T].
    """
    early = int(Ntn * 0.3)
    late = Ntn - early
    te = torch.rand(early, device=device) * (0.7 * T)
    tc = (torch.rand(late, device=device) * (0.3 * T)) + 0.7 * T
    return torch.cat([te, tc])


_TIME_CFG = _configure_problem_for_T(T=0.5, Nt_grape=500)
print(f"Configured temporal horizon: [0, {_TIME_CFG['T']:g}]")


# ════════════════════════════════════════════════════════════════
#  1. GRAPE  (faithful to grape_bueno.py — configurable T, seed=7, PyTorch GPU)
# ════════════════════════════════════════════════════════════════
print("\n"+"="*60)
print(f"GRAPE  (T={_TIME_CFG['T']:g}, seed=7, PyTorch GPU)")
print("="*60)

_Nb = 15; _Nt = _TIME_CFG["Nt_grape"]; _T = _TIME_CFG["T"]; _dt = _TIME_CFG["dt_grape"]
_n  = np.arange(1, _Nb+1)
_E  = (_n*np.pi)**2
_H0 = np.diag(_E)

def _xmel(m,n):
    if m==n: return 0.5
    p,q = m-n, m+n
    return ((-1)**p-1)/(p*np.pi)**2-((-1)**q-1)/(q*np.pi)**2

_V     = np.array([[_xmel(m+1,n+1) for n in range(_Nb)] for m in range(_Nb)])
_psi0  = np.zeros(_Nb, dtype=complex); _psi0[0]  = 1.0
_psif  = np.zeros(_Nb, dtype=complex); _psif[1]  = 1.0

_H0t   = torch.tensor(_H0,   dtype=torch.complex128, device=device)
_Vt    = torch.tensor(_V,    dtype=torch.complex128, device=device)
_psi0t = torch.tensor(_psi0, dtype=torch.complex128, device=device)
_psift = torch.tensor(_psif, dtype=torch.complex128, device=device)

def _fwd_grape(u):
    ut  = torch.tensor(u, dtype=torch.complex128, device=device)
    Hb  = _H0t.unsqueeze(0) + ut.view(-1,1,1)*_Vt.unsqueeze(0)
    ev,ec = torch.linalg.eigh(Hb)
    Ub  = ec @ (torch.exp(-1j*ev*_dt).unsqueeze(-1)*ec.mH)
    psi = _psi0t; traj = [psi.cpu().numpy()]
    for k in range(_Nt):
        psi = Ub[k]@psi; traj.append(psi.cpu().numpy())
    return traj

def _negF_grape(u_flat):
    ut  = torch.tensor(u_flat, dtype=torch.complex128, device=device)
    Hb  = _H0t.unsqueeze(0) + ut.view(-1,1,1)*_Vt.unsqueeze(0)
    ev,ec = torch.linalg.eigh(Hb)
    Ub  = ec @ (torch.exp(-1j*ev*_dt).unsqueeze(-1)*ec.mH)
    tr  = torch.zeros((_Nt+1,_Nb), dtype=torch.complex128, device=device)
    tr[0] = _psi0t; psi = _psi0t
    for k in range(_Nt):
        psi = Ub[k]@psi; tr[k+1] = psi
    lam  = torch.vdot(_psift, tr[-1]); F = torch.abs(lam)**2
    chi  = lam*_psift
    grad = torch.zeros(_Nt, dtype=torch.float64, device=device)
    for k in range(_Nt-1, -1, -1):
        grad[k] = 2*_dt*torch.imag(torch.vdot(chi, _Vt@tr[k]))
        chi = Ub[k].mH@chi
    return -F.item(), -grad.cpu().numpy()

_omega = _E[1]-_E[0]
_tmid  = _TIME_CFG["tmid_grape"]
_env   = np.exp(-0.5*((_tmid-_TIME_CFG["grape_env_center"])/_TIME_CFG["grape_env_sigma"])**2)
np.random.seed(7)
_u0 = 25.0*_env*np.sin(_omega*_tmid) + 0.8*np.random.randn(_Nt)

_log_g = []
def _negF_log(u):
    v,gr = _negF_grape(u); _log_g.append(-v); return v, gr

print("Optimising GRAPE …")
_t0    = time.time()
_res   = minimize(_negF_log, _u0, method="L-BFGS-B", jac=True, options={"maxiter":400})
print(f"Done in {time.time()-_t0:.1f} s")

u_g    = _res.x
F_g    = -_res.fun
traj_g = _fwd_grape(u_g)
print(f"GRAPE fidelity: {F_g:.7f}")

# Spatial grid
_Nx_g    = 700
x_g      = np.linspace(0, 1, _Nx_g)
_Phi_g   = np.sqrt(2)*np.sin(np.outer(_n, np.pi*x_g))
wf_g     = lambda c: c @ _Phi_g
phi1_g   = np.sqrt(2)*np.sin(np.pi*x_g)
phi2_g   = np.sqrt(2)*np.sin(2*np.pi*x_g)
t_g      = np.linspace(0, _T, _Nt+1)
norm_g   = np.array([np.real(traj_g[k].conj()@traj_g[k]) for k in range(_Nt+1)])
p0_g     = np.abs(wf_g(traj_g[0]))**2
pf_g     = np.abs(wf_g(traj_g[-1]))**2
u2_g     = u_g**2
ri_g     = _running_integral(_tmid, u2_g)

# ════════════════════════════════════════════════════════════════
#  2. MLP / PINN  (faithful to mlp_bueno.py — seed=42, 15 000 ep)
# ════════════════════════════════════════════════════════════════
print("\n"+"="*60)
print("MLP / PINN  (seed=42, 15 000 epochs)")
print("="*60)

def _set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    try: torch.use_deterministic_algorithms(True)
    except: pass
    os.environ["PYTHONHASHSEED"] = str(s)

_set_seed()

class _NLS_Net(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.act = nn.Tanh()
        self.L   = nn.ModuleList([nn.Linear(layers[i],layers[i+1])
                                   for i in range(len(layers)-1)])
    def forward(self, x):
        for i in range(len(self.L)-1): x = self.act(self.L[i](x))
        return self.L[-1](x)

class _PINN_QC:
    def __init__(self, T):
        self.T = T
        self.net_psi     = _NLS_Net([2,64,64,64,64,2]).to(device)
        self.net_control = _NLS_Net([1,64,64,64,64,1]).to(device)
        self.opt = torch.optim.Adam(
            list(self.net_psi.parameters())+list(self.net_control.parameters()), lr=1e-3)
    def get_psi(self, x, t):
        # Hard boundary ansatz: guarantees psi(0,t)=psi(1,t)=0 algebraically
        tau = t / self.T
        return self.net_psi(torch.cat([x,tau],1)) * (x*(1.0-x))
    def get_u(self, t):
        return self.net_control(t / self.T)
    def _net_f(self, x, t):
        x.requires_grad_(True); t.requires_grad_(True)
        psi = self.get_psi(x,t); R,I = psi[:,0:1],psi[:,1:2]
        u   = self.get_u(t)
        Rt  = torch.autograd.grad(R,  t,  torch.ones_like(R),  create_graph=True)[0]
        It  = torch.autograd.grad(I,  t,  torch.ones_like(I),  create_graph=True)[0]
        Rx  = torch.autograd.grad(R,  x,  torch.ones_like(R),  create_graph=True)[0]
        Rxx = torch.autograd.grad(Rx, x,  torch.ones_like(Rx), create_graph=True)[0]
        Ix  = torch.autograd.grad(I,  x,  torch.ones_like(I),  create_graph=True)[0]
        Ixx = torch.autograd.grad(Ix, x,  torch.ones_like(Ix), create_graph=True)[0]
        return It+Rxx+u*x*R, -Rt+Ixx+u*x*I
    def compute_loss(self, xf,tf,xi,ti,p1,xfin,tfin,p2,xn,tn,Ntn,Nxn):
        # x_b/t_b/loss_b removed: boundary is exactly enforced by the ansatz
        fR,fI = self._net_f(xf,tf);  Lf = torch.mean(fR**2+fI**2)
        pi    = self.get_psi(xi,ti)
        Li    = torch.mean((pi[:,0:1]-p1)**2+pi[:,1:2]**2)
        pf    = self.get_psi(xfin,tfin)
        Lfin  = torch.mean((pf[:,0:1]-p2)**2+pf[:,1:2]**2)
        pn    = self.get_psi(xn,tn)
        pr    = pn[:,0]**2+pn[:,1]**2
        integ = torch.sum(pr.view(Ntn,Nxn), dim=1)*(1.0/(Nxn-1))
        Ln    = torch.mean((integ-1)**2)
        return Lf + 100*Li + 200*Lfin + 5000*Ln

_T_m=_TIME_CFG["T"]; _Nf_m=5000; _Ni_m=1000; _Nfin_m=1000
# No intermediate re-seeds: RNG flows from _set_seed() above, exactly as mlp_bueno.py
xf_m  = torch.rand(_Nf_m,1).to(device).requires_grad_(True)
tf_m  = (torch.rand(_Nf_m,1)*_T_m).to(device).requires_grad_(True)
# x_b/t_b removed: boundary enforced exactly by the ansatz
xi_m  = torch.rand(_Ni_m,1).to(device); ti_m = torch.zeros(_Ni_m,1).to(device)
p1_m  = (math.sqrt(2)*torch.sin(math.pi*xi_m)).to(device)
xfin_m= torch.rand(_Nfin_m,1).to(device)
tfin_m= torch.ones(_Nfin_m,1).to(device)*_T_m
p2_m  = (math.sqrt(2)*torch.sin(2*math.pi*xfin_m)).to(device)
_Ntn_m,_Nxn_m = 400, 1000
_tv = _adaptive_norm_times(_T_m, _Ntn_m, device)
_xv = torch.linspace(0,1,_Nxn_m).to(device)
_Tg_m,_Xg_m = torch.meshgrid(_tv,_xv,indexing='ij')
xn_m = _Xg_m.reshape(-1,1); tn_m = _Tg_m.reshape(-1,1)

# Model constructed here: weights initialized from RNG state after all data above,
# exactly matching mlp_bueno.py (no _set_seed() reset before this line)
mlp = _PINN_QC(_T_m)
print("Training MLP (15 000 epochs) …")
for ep in range(15000):
    mlp.opt.zero_grad()
    l = mlp.compute_loss(xf_m,tf_m,xi_m,ti_m,p1_m,xfin_m,tfin_m,p2_m,
                         xn_m,tn_m,_Ntn_m,_Nxn_m)
    l.backward(); mlp.opt.step()
    if ep%1000==0: print(f"  ep {ep:5d}  loss={l.item():.5f}")

tp_m = torch.linspace(0,_T_m,500).view(-1,1).to(device)
xp_m = torch.linspace(0,1,200).view(-1,1).to(device)
with torch.no_grad():
    u_m_t   = mlp.get_u(tp_m)
    E_m     = torch.trapz(u_m_t.squeeze()**2, tp_m.squeeze())
    ps0_m   = mlp.get_psi(xp_m, torch.zeros_like(xp_m))
    psf_m   = mlp.get_psi(xp_m, torch.ones_like(xp_m)*_T_m)
    pr0_m   = ps0_m[:,0]**2+ps0_m[:,1]**2
    prf_m   = psf_m[:,0]**2+psf_m[:,1]**2

xp_m_np = xp_m.cpu().numpy().flatten(); tp_m_np = tp_m.cpu().numpy().flatten()
u_m_np  = u_m_t.cpu().numpy().flatten()
pr0_m_np= pr0_m.cpu().numpy().flatten(); prf_m_np= prf_m.cpu().numpy().flatten()
phi1_m  = (math.sqrt(2)*np.sin(math.pi*xp_m_np))**2
phi2_m  = (math.sqrt(2)*np.sin(2*math.pi*xp_m_np))**2
u2_m    = u_m_np**2; ri_m = _running_integral(tp_m_np, u2_m)

print("Computing MLP norm …")
dx_m = 1/(len(xp_m)-1); norm_m = []
with torch.no_grad():
    for tv in tp_m_np:
        tl = torch.ones_like(xp_m)*tv
        ps = mlp.get_psi(xp_m, tl)
        norm_m.append((torch.sum(ps[:,0]**2+ps[:,1]**2)*dx_m).item())

# ════════════════════════════════════════════════════════════════
#  3. KAN  (faithful to kan_bueno.py — seed=42, 10 000 epochs)
# ════════════════════════════════════════════════════════════════
print("\n"+"="*60)
print("KAN  (seed=42, 10 000 epochs)")
print("="*60)

_set_seed()
torch.cuda.empty_cache()

class _FourierKAN(nn.Module):
    def __init__(self, i, o, F=4):
        super().__init__()
        self.F  = F
        self.cs = nn.Parameter(torch.randn(o,i,F)*.02)
        self.cc = nn.Parameter(torch.randn(o,i,F)*.02)
        self.base = nn.Linear(i,o); self.act = nn.SiLU()
    def forward(self, x):
        s  = self.base(self.act(x)); xb = torch.tanh(x)
        for k in range(1, self.F+1):
            s += torch.einsum('oi,bi->bo', self.cs[:,:,k-1], torch.sin(math.pi*k*xb))
            s += torch.einsum('oi,bi->bo', self.cc[:,:,k-1], torch.cos(math.pi*k*xb))
        return s

class _KAN_Net(nn.Module):
    def __init__(self, layers, F=4):
        super().__init__()
        self.L = nn.ModuleList([_FourierKAN(layers[i],layers[i+1],F)
                                 for i in range(len(layers)-1)])
    def forward(self, x):
        for l in self.L:
            x = l(x)
        return x

class _KAN_QC:
    def __init__(self, T):
        self.T = T
        self.net_psi     = _KAN_Net([2,32,32,2]).to(device)
        self.net_control = _KAN_Net([1,16,16,1]).to(device)
        params = list(self.net_psi.parameters())+list(self.net_control.parameters())
        self.opt = torch.optim.Adam(params, lr=1e-3)
        self.sch = torch.optim.lr_scheduler.StepLR(self.opt, step_size=2500, gamma=0.5)
    def get_psi(self, x, t):
        tau = t / self.T
        return self.net_psi(torch.cat([x,tau],1))*(x*(1-x))   # boundary ansatz
    def get_u(self, t):
        return self.net_control(t / self.T)
    def _net_f(self, x, t):
        x.requires_grad_(True); t.requires_grad_(True)
        psi = self.get_psi(x,t); R,I = psi[:,0:1],psi[:,1:2]
        u   = self.get_u(t)
        Rt  = torch.autograd.grad(R,  t,  torch.ones_like(R),  create_graph=True)[0]
        It  = torch.autograd.grad(I,  t,  torch.ones_like(I),  create_graph=True)[0]
        Rx  = torch.autograd.grad(R,  x,  torch.ones_like(R),  create_graph=True)[0]
        Rxx = torch.autograd.grad(Rx, x,  torch.ones_like(Rx), create_graph=True)[0]
        Ix  = torch.autograd.grad(I,  x,  torch.ones_like(I),  create_graph=True)[0]
        Ixx = torch.autograd.grad(Ix, x,  torch.ones_like(Ix), create_graph=True)[0]
        return It+Rxx+u*x*R, -Rt+Ixx+u*x*I
    def compute_loss(self, xf,tf,xi,ti,p1,xfin,tfin,p2,xn,tn,Ntn,Nxn):
        fR,fI = self._net_f(xf,tf); Lf = torch.mean(fR**2+fI**2)
        pi = self.get_psi(xi,ti); Li = torch.mean((pi[:,0:1]-p1)**2+pi[:,1:2]**2)
        pf = self.get_psi(xfin,tfin); Lfin = torch.mean((pf[:,0:1]-p2)**2+pf[:,1:2]**2)
        pn  = self.get_psi(xn,tn); pr = pn[:,0]**2+pn[:,1]**2
        integ = torch.sum(pr.view(Ntn,Nxn),dim=1)*(1.0/(Nxn-1))
        Ln = torch.mean((integ-1)**2)
        return Lf + 250*Li + 200*Lfin + 2500*Ln

_T_k=_TIME_CFG["T"]; _Nf_k=6000; _Ni_k=1000; _Nfin_k=1000
xf_k  = torch.rand(_Nf_k,1).to(device).requires_grad_(True)
tf_k  = (torch.rand(_Nf_k,1)*_T_k).to(device).requires_grad_(True)
xi_k  = torch.rand(_Ni_k,1).to(device); ti_k = torch.zeros(_Ni_k,1).to(device)
p1_k  = (math.sqrt(2)*torch.sin(math.pi*xi_k)).to(device)
xfin_k= torch.rand(_Nfin_k,1).to(device); tfin_k=torch.ones(_Nfin_k,1).to(device)*_T_k
p2_k  = (math.sqrt(2)*torch.sin(2*math.pi*xfin_k)).to(device)
_Ntn_k,_Nxn_k = 400,1000
_tv_k = _adaptive_norm_times(_T_k, _Ntn_k, device)
_xv_k = torch.linspace(0,1,_Nxn_k).to(device)
_Tg_k,_Xg_k = torch.meshgrid(_tv_k,_xv_k,indexing='ij')
xn_k = _Xg_k.reshape(-1,1); tn_k = _Tg_k.reshape(-1,1)

kan = _KAN_QC(_T_k)
print("Training KAN (10 000 epochs) …")
for ep in range(10000):
    kan.opt.zero_grad()
    l = kan.compute_loss(xf_k,tf_k,xi_k,ti_k,p1_k,xfin_k,tfin_k,p2_k,
                         xn_k,tn_k,_Ntn_k,_Nxn_k)
    l.backward(); kan.opt.step(); kan.sch.step()
    if ep%1000==0: print(f"  ep {ep:5d}  loss={l.item():.5f}  lr={kan.sch.get_last_lr()[0]:.5f}")

tp_k = torch.linspace(0,_T_k,500).view(-1,1).to(device)
xp_k = torch.linspace(0,1,200).view(-1,1).to(device)
with torch.no_grad():
    u_k_t  = kan.get_u(tp_k)
    E_k    = torch.sum(u_k_t**2)*(_T_k/(len(tp_k)-1))
    ps0_k  = kan.get_psi(xp_k, torch.zeros_like(xp_k))
    psf_k  = kan.get_psi(xp_k, torch.ones_like(xp_k)*_T_k)
    pr0_k  = ps0_k[:,0]**2+ps0_k[:,1]**2
    prf_k  = psf_k[:,0]**2+psf_k[:,1]**2

xp_k_np = xp_k.cpu().numpy().flatten(); tp_k_np = tp_k.cpu().numpy().flatten()
u_k_np  = u_k_t.cpu().numpy().flatten()
pr0_k_np= pr0_k.cpu().numpy().flatten(); prf_k_np= prf_k.cpu().numpy().flatten()
phi1_k  = (math.sqrt(2)*np.sin(math.pi*xp_k_np))**2
phi2_k  = (math.sqrt(2)*np.sin(2*math.pi*xp_k_np))**2
u2_k    = u_k_np**2; ri_k = _running_integral(tp_k_np, u2_k)

print("Computing KAN norm …")
dx_k = 1/(len(xp_k)-1); norm_k = []
with torch.no_grad():
    for tv in tp_k_np:
        tl = torch.ones_like(xp_k)*tv
        ps = kan.get_psi(xp_k,tl)
        norm_k.append((torch.sum(ps[:,0]**2+ps[:,1]**2)*dx_k).item())

# ════════════════════════════════════════════════════════════════
#  4. HYBRID MLP-KAN  (faithful to mlp_kan.py — seed=42, 4 000 ep)
# ════════════════════════════════════════════════════════════════
print("\n"+"="*60)
print("Hybrid MLP-KAN  (seed=42, 4 000 epochs)")
print("="*60)

_set_seed()

class _KANLayer_H(nn.Module):
    def __init__(self, i, o, F=4):
        super().__init__()
        self.F  = F
        self.sn = nn.Parameter(torch.randn(o,i,F)*.02)
        self.cn = nn.Parameter(torch.randn(o,i,F)*.02)
        self.base = nn.Linear(i,o)
    def forward(self, x):
        xt = torch.tanh(x); out = self.base(xt)
        for k in range(1, self.F+1):
            out += torch.einsum('oi,bi->bo', self.sn[:,:,k-1], torch.sin(math.pi*k*xt))
            out += torch.einsum('oi,bi->bo', self.cn[:,:,k-1], torch.cos(math.pi*k*xt))
        return out

class _KAN_H(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.L = nn.ModuleList([_KANLayer_H(layers[i],layers[i+1])
                                 for i in range(len(layers)-1)])
    def forward(self, x):
        for l in self.L: x=l(x)
        return x

class _PINN_H(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.act = nn.Tanh()
        self.L   = nn.ModuleList([nn.Linear(layers[i],layers[i+1])
                                   for i in range(len(layers)-1)])
    def forward(self, x):
        for i in range(len(self.L)-1): x = self.act(self.L[i](x))
        return self.L[-1](x)

class _Hybrid(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T
        self.kan_psi  = _KAN_H([2,32,32,2]); self.pinn_psi  = _PINN_H([2,64,64,2])
        self.kan_u    = _KAN_H([1,16,16,1]); self.pinn_u    = _PINN_H([1,64,64,1])
        self.wk = nn.Parameter(torch.tensor(1.0))
        self.wp = nn.Parameter(torch.tensor(1.0))
    def psi(self, x, t):
        tau = t / self.T
        xt = torch.cat([x,tau],1)
        psi = self.wk*self.kan_psi(xt) + self.wp*self.pinn_psi(xt)
        # Hard boundary ansatz: guarantees psi(0,t)=psi(1,t)=0 algebraically
        return psi * (x * (1.0 - x))
    def u(self, t):
        tau = t / self.T
        return self.wk*self.kan_u(tau) + self.wp*self.pinn_u(tau)

def _loss_hybrid(model, xf,tf, xi,ti,phi_i, xfi,tfi,phi_f, xn,tn, Ntn,Nxn):
    xf = xf.clone().detach().requires_grad_(True)
    tf = tf.clone().detach().requires_grad_(True)
    psi = model.psi(xf,tf); R,I = psi[:,0:1], psi[:,1:2]; u = model.u(tf)
    Rt  = torch.autograd.grad(R, tf, torch.ones_like(R),  create_graph=True)[0]
    It  = torch.autograd.grad(I, tf, torch.ones_like(I),  create_graph=True)[0]
    Rx  = torch.autograd.grad(R, xf, torch.ones_like(R),  create_graph=True)[0]
    Rxx = torch.autograd.grad(Rx,xf, torch.ones_like(Rx), create_graph=True)[0]
    Ix  = torch.autograd.grad(I, xf, torch.ones_like(I),  create_graph=True)[0]
    Ixx = torch.autograd.grad(Ix,xf, torch.ones_like(Ix), create_graph=True)[0]
    fR = It+Rxx+u*xf*R; fI = -Rt+Ixx+u*xf*I
    Lf = torch.mean(fR**2+fI**2)
    pi  = model.psi(xi,ti);   Li  = torch.mean((pi[:,0:1]-phi_i)**2+pi[:,1:2]**2)
    pf  = model.psi(xfi,tfi); Lfi = torch.mean((pf[:,0:1]-phi_f)**2+pf[:,1:2]**2)
    pn  = model.psi(xn,tn);   pr  = pn[:,0:1]**2+pn[:,1:2]**2
    pr  = pr.view(Ntn,Nxn)
    Ln  = torch.mean((torch.sum(pr,dim=1)*(1.0/(Nxn-1))-1)**2)
    # norm weight raised to 5000 (same as MLP) + adaptive grid below
    return Lf + 100*Li + 200*Lfi + 5000*Ln

_T_h=_TIME_CFG["T"]
torch.manual_seed(SEED); torch.cuda.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
xf_h  = torch.rand(3000,1).to(device); tf_h = torch.rand(3000,1).to(device)*_T_h
torch.manual_seed(SEED+1); torch.cuda.manual_seed(SEED+1)
xi_h  = torch.rand(1000,1).to(device); ti_h = torch.zeros_like(xi_h)
phi_i_h = torch.sqrt(torch.tensor(2.))*torch.sin(math.pi*xi_h)
torch.manual_seed(SEED+2); torch.cuda.manual_seed(SEED+2)
xfi_h = torch.rand(1000,1).to(device); tfi_h = torch.ones_like(xfi_h)*_T_h
phi_f_h = torch.sqrt(torch.tensor(2.))*torch.sin(2*math.pi*xfi_h)
# Adaptive norm grid: same strategy as in the original code, scaled to [0,T].
_Ntn_h, _Nxn_h = 400, 1000
_tv_h = _adaptive_norm_times(_T_h, _Ntn_h, device)
_xv_h = torch.linspace(0,1,_Nxn_h).to(device)
_Th2,_Xh2 = torch.meshgrid(_tv_h,_xv_h,indexing='ij')
xn_h = _Xh2.reshape(-1,1); tn_h = _Th2.reshape(-1,1)

_set_seed()
hybrid  = _Hybrid(_T_h).to(device)
opt_hyb = torch.optim.Adam(hybrid.parameters(), lr=1e-3)
print("Training Hybrid (4 000 epochs) …")
for ep in range(4000):
    opt_hyb.zero_grad()
    l = _loss_hybrid(hybrid, xf_h,tf_h, xi_h,ti_h,phi_i_h,
                     xfi_h,tfi_h,phi_f_h, xn_h,tn_h, _Ntn_h,_Nxn_h)
    l.backward(); opt_hyb.step()
    if ep%500==0: print(f"  ep {ep:5d}  loss={l.item():.5f}")

tp_h = torch.linspace(0,_T_h,500).view(-1,1).to(device)
xp_h = torch.linspace(0,1,200).view(-1,1).to(device)
with torch.no_grad():
    u_h_t  = hybrid.u(tp_h)
    E_h    = torch.trapz(u_h_t.squeeze()**2, tp_h.squeeze())
    ps0_h  = hybrid.psi(xp_h, torch.zeros_like(xp_h))
    psf_h  = hybrid.psi(xp_h, torch.ones_like(xp_h)*_T_h)
    pr0_h  = ps0_h[:,0]**2+ps0_h[:,1]**2
    prf_h  = psf_h[:,0]**2+psf_h[:,1]**2

xp_h_np = xp_h.cpu().numpy().flatten(); tp_h_np = tp_h.cpu().numpy().flatten()
u_h_np  = u_h_t.cpu().numpy().flatten()
pr0_h_np= pr0_h.cpu().numpy().flatten(); prf_h_np= prf_h.cpu().numpy().flatten()
phi1_h  = (math.sqrt(2)*np.sin(math.pi*xp_h_np))**2
phi2_h  = (math.sqrt(2)*np.sin(2*math.pi*xp_h_np))**2
u2_h    = u_h_np**2; ri_h = _running_integral(tp_h_np, u2_h)

print("Computing Hybrid norm …")
dx_h = 1/(len(xp_h)-1); norm_h = []
with torch.no_grad():
    for tv in tp_h_np:
        tl = torch.ones_like(xp_h)*tv
        ps = hybrid.psi(xp_h,tl)
        norm_h.append((torch.sum(ps[:,0]**2+ps[:,1]**2)*dx_h).item())

# ════════════════════════════════════════════════════════════════
#  COMPARISON FIGURES
# ════════════════════════════════════════════════════════════════
print("\n"+"="*60+"\nGENERATING COMPARISON FIGURES\n"+"="*60)

_TIME_XLIM = (0, _TIME_CFG["T"])
_NORM_YLIM = (0.95, 1.05)
_T_TEXT = f"{_TIME_CFG['T']:g}"

_METHODS = [
    ("GRAPE",  x_g,     p0_g,     pf_g,     "blue",       "darkgreen"),
    ("MLP",    xp_m_np, pr0_m_np, prf_m_np, "forestgreen","purple"),
    ("KAN",    xp_k_np, pr0_k_np, prf_k_np, "darkorange", "darkorange"),
    ("Hybrid", xp_h_np, pr0_h_np, prf_h_np, "red",        "darkorchid"),
]

# ── Fig 01: Initial states 2×2 ──────────────────────────────────
fig,axes = plt.subplots(2,2,figsize=(14,10))
fig.suptitle(r"Initial State $|\psi(x,0)|^2$ — All Methods",fontsize=14,fontweight='bold')
for ax,(name,x_,p0_,_pf,col_,_) in zip(axes.flat, _METHODS):
    ax.plot(x_,p0_,color=col_,lw=2.4,label=f"{name} (t=0)")
    ax.plot(x_g,phi1_g**2,"--",color="black",lw=1.5,label="Exact Level 1")
    ax.set_title(f"{name} — Initial State"); ax.set_xlabel("x"); ax.set_ylabel(r"$|\psi|^2$")
    ax.set_xlim(0,1); ax.legend(); ax.grid(True)
plt.tight_layout(); plt.savefig(f"{OUT}/fig01_initial_states_2x2.png",dpi=150); plt.show()
print("Saved fig01_initial_states_2x2.png")

# ── Fig 02: Final states 2×2 ────────────────────────────────────
fig,axes = plt.subplots(2,2,figsize=(14,10))
fig.suptitle(rf"Final State $|\psi(x,{_T_TEXT})|^2$ — All Methods",fontsize=14,fontweight='bold')
for ax,(name,x_,_p0,pf_,col_,_) in zip(axes.flat, _METHODS):
    extra = f" (F={F_g:.5f})" if name=="GRAPE" else ""
    ax.plot(x_,pf_,color=col_,lw=2.4,label=f"{name} (t={_T_TEXT}){extra}")
    ax.plot(x_g,phi2_g**2,"--",color="black",lw=1.5,label="Exact Level 2")
    ax.set_title(f"{name} — Final State"); ax.set_xlabel("x"); ax.set_ylabel(r"$|\psi|^2$")
    ax.set_xlim(0,1); ax.legend(); ax.grid(True)
plt.tight_layout(); plt.savefig(f"{OUT}/fig02_final_states_2x2.png",dpi=150); plt.show()
print("Saved fig02_final_states_2x2.png")

# ── Fig 03: All methods on same axes (initial + final) ──────────
_LS = ["-","-.",":",  (0,(3,1,1,1))]
fig,axes = plt.subplots(1,2,figsize=(14,5))
fig.suptitle("All Methods: Initial (left) and Final (right)",fontsize=13,fontweight='bold')
for (name,x_,p0_,pf_,col_,_),ls in zip(_METHODS,_LS):
    axes[0].plot(x_,p0_,color=col_,lw=2,ls=ls,label=name)
    axes[1].plot(x_,pf_,color=col_,lw=2,ls=ls,
                 label=name+(f" F={F_g:.4f}" if name=="GRAPE" else ""))
axes[0].plot(x_g,phi1_g**2,"--",color="black",lw=1.5,label="Exact Lvl 1")
axes[1].plot(x_g,phi2_g**2,"--",color="black",lw=1.5,label="Exact Lvl 2")
for ax,ttl in zip(axes,["Initial (t=0)",f"Final (t={_T_TEXT})"]):
    ax.set_title(ttl); ax.set_xlabel("x"); ax.set_ylabel(r"$|\psi|^2$")
    ax.set_xlim(0,1); ax.legend(); ax.grid(True)
plt.tight_layout(); plt.savefig(f"{OUT}/fig03_states_all_methods.png",dpi=150); plt.show()
print("Saved fig03_states_all_methods.png")

# ── Fig 04: Norm — all in one ────────────────────────────────────
_norm_data = [
    ("GRAPE",  t_g,     norm_g, "blue",       "-"),
    ("MLP",    tp_m_np, norm_m, "forestgreen","-."),
    ("KAN",    tp_k_np, norm_k, "darkorange", ":"),
    ("Hybrid", tp_h_np, norm_h, "red",        (0,(3,1,1,1))),
]
fig,ax = plt.subplots(figsize=(10,5))
for name,t_,n_,col_,ls_ in _norm_data:
    ax.plot(t_,n_,color=col_,lw=2,ls=ls_,label=name)
ax.axhline(1,color="black",ls="--",lw=1.5,label="Exact = 1")
ax.set_ylim(*_NORM_YLIM); ax.set_xlim(*_TIME_XLIM)
ax.set_title("Norm Conservation — All Methods"); ax.set_xlabel("t"); ax.set_ylabel("Norm")
ax.legend(); ax.grid(True)
plt.tight_layout(); plt.savefig(f"{OUT}/fig04_norm_all_in_one.png",dpi=150); plt.show()
print("Saved fig04_norm_all_in_one.png")

# ── Fig 05: Norm — 2×2 subplots ─────────────────────────────────
fig,axes = plt.subplots(2,2,figsize=(14,10))
fig.suptitle("Norm Conservation per Method",fontsize=14,fontweight='bold')
for ax,(name,t_,n_,col_,_) in zip(axes.flat, _norm_data):
    ax.plot(t_,n_,color=col_,lw=2,label=name)
    ax.axhline(1,color="black",ls="--",alpha=0.5)
    ax.set_ylim(*_NORM_YLIM); ax.set_xlim(*_TIME_XLIM)
    ax.set_title(f"{name} — Norm Conservation"); ax.set_xlabel("t"); ax.set_ylabel("Norm")
    ax.legend(); ax.grid(True)
plt.tight_layout(); plt.savefig(f"{OUT}/fig05_norm_subplots_2x2.png",dpi=150); plt.show()
print("Saved fig05_norm_subplots_2x2.png")

# ── GIF Norm: ||psi(t)||² — all methods, same scale ─────────────
print("\n"+"="*60)
print("Generating norm GIF — all methods")
print("="*60)

_N_norm = 160
_fps_norm = 6
_pause_norm = 45
_frames_norm = list(range(_N_norm)) + [_N_norm-1]*_pause_norm
_norm_txt_y = [0.97, 0.83, 0.69, 0.55]

fig_norm, ax_norm = plt.subplots(figsize=(12,6))
fig_norm.patch.set_facecolor('white')
ax_norm.set_facecolor('white')
for name,t_,n_,col_,ls_ in _norm_data:
    ax_norm.plot(t_, n_, color=col_, lw=1.4, ls=ls_, alpha=0.18)
ax_norm.axhline(1.0, color="black", ls="--", lw=1.3, label="Exact = 1")
ax_norm.set_xlim(*_TIME_XLIM)
ax_norm.set_ylim(*_NORM_YLIM)
ax_norm.set_title(r"Norm of $\psi$ — All Methods", fontsize=13, fontweight='bold')
ax_norm.set_xlabel("$t$", fontsize=12)
ax_norm.set_ylabel(r"$\Vert \psi(t) \Vert^2$", fontsize=12)
ax_norm.grid(True, linestyle='--', alpha=0.55)

_norm_lines = []
_norm_dots = []
_norm_texts = []
for idx, (name,t_,n_,col_,ls_) in enumerate(_norm_data):
    ln, = ax_norm.plot([], [], color=col_, lw=2.2, ls=ls_, label=name)
    dot, = ax_norm.plot([], [], 'o', color=col_, ms=6, zorder=5)
    txt = ax_norm.text(0.985, _norm_txt_y[idx], '', transform=ax_norm.transAxes,
                       ha='right', va='top', fontsize=11, fontweight='bold',
                       color=col_,
                       bbox=dict(fc='white', ec=col_, boxstyle='round,pad=0.3',
                                 alpha=0.92, lw=1.2))
    _norm_lines.append(ln); _norm_dots.append(dot); _norm_texts.append(txt)

ax_norm.legend(loc='upper left', fontsize=10, framealpha=0.9)
plt.tight_layout()

def _update_gif_norm(frame):
    for idx, (name, t_, n_, col_, ls_) in enumerate(_norm_data):
        k = min(int(frame * (len(t_)-1) / (_N_norm-1)), len(t_)-1)
        _norm_lines[idx].set_data(t_[:k+1], n_[:k+1])
        _norm_dots[idx].set_data([t_[k]], [n_[k]])
        _norm_texts[idx].set_text(f"{name}: $t={t_[k]:.3f}$  norm={n_[k]:.4f}")
    return _norm_lines + _norm_dots + _norm_texts

print("  Rendering GIF …")
_anim_norm = FuncAnimation(fig_norm, _update_gif_norm, frames=_frames_norm,
                           interval=int(1000/_fps_norm), blit=True)
_anim_norm.save(f"{OUT}/gif_norm_psi_4methods.gif",
                writer=PillowWriter(fps=_fps_norm))
plt.close()
print("Saved gif_norm_psi_4methods.gif")

# ── Fig 06: u(t) — 2×2, SHARED y-scale ──────────────────────────
_u_data = [
    ("GRAPE",  _tmid,   u_g,    "darkgreen"),
    ("MLP",    tp_m_np, u_m_np, "purple"),
    ("KAN",    tp_k_np, u_k_np, "darkorange"),
    ("Hybrid", tp_h_np, u_h_np, "darkorchid"),
]
_u_ymin = min(u_g.min(), u_m_np.min(), u_k_np.min(), u_h_np.min())
_u_ymax = max(u_g.max(), u_m_np.max(), u_k_np.max(), u_h_np.max())
_u_pad  = 0.05*max(_u_ymax-_u_ymin, 1e-12)
_u_ylim = (_u_ymin-_u_pad, _u_ymax+_u_pad)
fig,axes = plt.subplots(2,2,figsize=(14,10))
fig.suptitle("Control Signal $u(t)$ — All Methods (same scale)",fontsize=14,fontweight='bold')
for ax,(name,t_,u_,col_) in zip(axes.flat, _u_data):
    ax.plot(t_,u_,color=col_,lw=1.5)
    ax.set_title(f"{name} — $u(t)$"); ax.set_xlabel("t"); ax.set_ylabel("$u(t)$")
    ax.set_xlim(*_TIME_XLIM); ax.set_ylim(_u_ylim); ax.grid(True)
plt.tight_layout(); plt.savefig(f"{OUT}/fig06_u_subplots_2x2.png",dpi=150); plt.show()
print("Saved fig06_u_subplots_2x2.png")

# ── Fig 07: |u(t)|² — 2×2, SHARED y-scale ──────────────────────
_u2_data = [
    ("GRAPE",  _tmid,   u2_g, ri_g, "darkgreen"),
    ("MLP",    tp_m_np, u2_m, ri_m, "purple"),
    ("KAN",    tp_k_np, u2_k, ri_k, "darkorange"),
    ("Hybrid", tp_h_np, u2_h, ri_h, "darkorchid"),
]
# Shared y-limit so visual comparison across methods is fair
_u2_ymax = max(u2_g.max(), u2_m.max(), u2_k.max(), u2_h.max()) * 1.15
_u2_ylim = (-_u2_ymax * 0.03, _u2_ymax)
fig,axes = plt.subplots(2,2,figsize=(14,10))
fig.suptitle(r"$|u(t)|^2$ — All Methods (same scale)",
             fontsize=14,fontweight='bold')
for ax,(name,t_,u2_,ri_,col_) in zip(axes.flat, _u2_data):
    ax.plot(t_,u2_,color=col_,lw=1.5)
    ax.fill_between(t_,u2_,color=col_,alpha=0.28)
    ax.set_title(f"{name}  (total $E = {ri_[-1]:.2f}$)")
    ax.set_xlabel("t"); ax.set_ylabel(r"$|u(t)|^2$")
    ax.set_xlim(*_TIME_XLIM); ax.set_ylim(_u2_ylim); ax.grid(True)
plt.tight_layout(); plt.savefig(f"{OUT}/fig07_u2_subplots_2x2.png",dpi=150); plt.show()
print("Saved fig07_u2_subplots_2x2.png")

# ── Fig 08: |u(t)|² — all in one ────────────────────────────────
fig,ax = plt.subplots(figsize=(12,5))
for name,t_,u2_,ri_,col_ in _u2_data:
    ax.plot(t_,u2_,color=col_,lw=2,label=f"{name}  ($E={ri_[-1]:.2f}$)")
ax.set_title(r"$|u(t)|^2$ — All Methods"); ax.set_xlabel("t"); ax.set_ylabel(r"$|u(t)|^2$")
ax.set_xlim(*_TIME_XLIM); ax.legend(); ax.grid(True)
plt.tight_layout(); plt.savefig(f"{OUT}/fig08_u2_all_in_one.png",dpi=150); plt.show()
print("Saved fig08_u2_all_in_one.png")

# ── Fig 09: f(t) = ∫₀ᵗ|u|² ds — all methods, same scale ────────
fig,ax = plt.subplots(figsize=(12,5))
fig.suptitle(r"Accumulated Control Energy $f(t)=\int_0^t |u(s)|^2\,ds$ — All Methods",
             fontsize=13, fontweight='bold')
_fi_ls = ["-","-.",":", (0,(3,1,1,1))]
for (name,t_,u2_,ri_,col_),ls in zip(_u2_data, _fi_ls):
    ax.plot(t_, ri_, color=col_, lw=2.2, ls=ls,
            label=f"{name}  $f(T)={ri_[-1]:.2f}$")
ax.set_xlim(*_TIME_XLIM); ax.set_ylim(bottom=0)
ax.set_xlabel("$t$", fontsize=12); ax.set_ylabel(r"$f(t)=\int_0^t |u|^2\,ds$", fontsize=12)
ax.legend(fontsize=11); ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout(); plt.savefig(f"{OUT}/fig09_f_t_all_methods.png", dpi=150); plt.show()
print("Saved fig09_f_t_all_methods.png")

# ── Fig 10: f(t) — 2×2 subplots, same y-scale ───────────────────
_fi_ymax = max(ri_g[-1], ri_m[-1], ri_k[-1], ri_h[-1]) * 1.12
fig,axes = plt.subplots(2,2,figsize=(14,10))
fig.suptitle(r"Accumulated Control Energy $f(t)=\int_0^t |u(s)|^2\,ds$ — Same Scale",
             fontsize=13, fontweight='bold')
for ax,(name,t_,u2_,ri_,col_) in zip(axes.flat, _u2_data):
    ax.plot(t_, ri_, color=col_, lw=2.2)
    ax.fill_between(t_, ri_, color=col_, alpha=0.18)
    ax.set_title(f"{name}  $f(T)={ri_[-1]:.2f}$", fontsize=12)
    ax.set_xlabel("$t$", fontsize=11); ax.set_ylabel(r"$f(t)$", fontsize=11)
    ax.set_xlim(*_TIME_XLIM); ax.set_ylim(0, _fi_ymax); ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout(); plt.savefig(f"{OUT}/fig10_f_t_subplots_2x2.png", dpi=150); plt.show()
print("Saved fig10_f_t_subplots_2x2.png")

# ════════════════════════════════════════════════════════════════
#  GIF A — Wave evolution  2×2  (GRAPE | MLP | KAN | Hybrid)
# ════════════════════════════════════════════════════════════════
print("\n"+"="*60+"\nGenerating wave-evolution GIF (2×2)\n"+"="*60)

_Nw = 120; _fps_w = 10; _pause_w = 30
_t_gif = np.linspace(0, _T, _Nw)
_gi_g  = np.linspace(0, _Nt, _Nw, dtype=int)

print("  Precomputing MLP wave …")
_prob_w_m = []
with torch.no_grad():
    for tv in _t_gif:
        tl=torch.ones_like(xp_m)*tv
        ps=mlp.get_psi(xp_m, tl)
        _prob_w_m.append((ps[:,0]**2+ps[:,1]**2).cpu().numpy().flatten())

print("  Precomputing KAN wave …")
_prob_w_k = []
with torch.no_grad():
    for tv in _t_gif:
        tl=torch.ones_like(xp_k)*tv
        ps=kan.get_psi(xp_k,tl)
        _prob_w_k.append((ps[:,0]**2+ps[:,1]**2).cpu().numpy().flatten())

print("  Precomputing Hybrid wave …")
_prob_w_h = []
with torch.no_grad():
    for tv in _t_gif:
        tl=torch.ones_like(xp_h)*tv
        ps=hybrid.psi(xp_h,tl)
        _prob_w_h.append((ps[:,0]**2+ps[:,1]**2).cpu().numpy().flatten())

_frames_w = list(range(_Nw)) + [_Nw-1]*_pause_w

fig_w,axes_w = plt.subplots(2,2,figsize=(14,10))
fig_w.suptitle(r"Wave Evolution $|\psi(x,t)|^2$ — All Methods",
               fontsize=14,fontweight='bold')
fig_w.patch.set_facecolor('white')
for ax in axes_w.flat: ax.set_facecolor('white')

_ax_gw,_ax_mw,_ax_kw,_ax_hw = axes_w[0,0],axes_w[0,1],axes_w[1,0],axes_w[1,1]

for ax,lbl,ph1,ph2,x_,ylim in [
    (_ax_gw,"GRAPE",  phi1_g**2,phi2_g**2,x_g,    2.6),
    (_ax_mw,"MLP",    phi1_m,   phi2_m,   xp_m_np,2.5),
    (_ax_kw,"KAN",    phi1_k,   phi2_k,   xp_k_np,2.5),
    (_ax_hw,"Hybrid", phi1_h,   phi2_h,   xp_h_np,2.5),
]:
    ax.plot(x_,ph1,":",color="blue",alpha=0.4,label="Level 1")
    ax.plot(x_,ph2,":",color="red", alpha=0.4,label="Level 2 (target)")
    ax.set_xlim(0,1); ax.set_ylim(0,ylim)
    ax.set_title(f"{lbl} — Wave Evolution")
    ax.set_xlabel("x"); ax.set_ylabel(r"$|\psi(x,t)|^2$")
    ax.legend(loc="upper right"); ax.grid(True)

_ln_g,=_ax_gw.plot([],[],color="green",   lw=3,label=r"GRAPE $|\psi|^2$")
_ln_m,=_ax_mw.plot([],[],color="purple",  lw=3,label=r"MLP $|\psi|^2$")
_ln_k,=_ax_kw.plot([],[],color="darkorange",lw=3,label=r"KAN $|\psi|^2$")
_ln_h,=_ax_hw.plot([],[],color="red",     lw=3,label=r"Hybrid $|\psi|^2$")

_tx_g=_ax_gw.text(0.05,0.88,"",transform=_ax_gw.transAxes,fontweight='bold',fontsize=10)
_tx_m=_ax_mw.text(0.05,0.88,"",transform=_ax_mw.transAxes,fontweight='bold',fontsize=10)
_tx_k=_ax_kw.text(0.05,0.88,"",transform=_ax_kw.transAxes,fontweight='bold',fontsize=10)
_tx_h=_ax_hw.text(0.05,0.88,"",transform=_ax_hw.transAxes,fontweight='bold',fontsize=10)
plt.tight_layout()

def _update_wave(frame):
    kg = _gi_g[frame]; tc = _t_gif[frame]
    pg = np.abs(wf_g(traj_g[kg]))**2
    _ln_g.set_data(x_g, pg)
    fg = abs(_psif.conj()@traj_g[kg])**2
    _tx_g.set_text(f"t={tc:.2f}  F={fg:.4f}")
    _ln_m.set_data(xp_m_np, _prob_w_m[frame]); _tx_m.set_text(f"t={tc:.2f}")
    _ln_k.set_data(xp_k_np, _prob_w_k[frame]); _tx_k.set_text(f"t={tc:.2f}")
    _ln_h.set_data(xp_h_np, _prob_w_h[frame]); _tx_h.set_text(f"t={tc:.2f}")
    return _ln_g,_tx_g,_ln_m,_tx_m,_ln_k,_tx_k,_ln_h,_tx_h

print("  Rendering GIF …")
_anim_w = FuncAnimation(fig_w, _update_wave, frames=_frames_w, interval=100, blit=True)
_anim_w.save(f"{OUT}/gif_wave_evolution_4methods.gif", writer=PillowWriter(fps=_fps_w))
plt.close()
print("Saved gif_wave_evolution_4methods.gif")

# ════════════════════════════════════════════════════════════════
#  GIF B — Fourier coefficient bar chart (all 4 methods, 2×2)
#           Bars: |c_n|² normalised so sum = 1  (n = 1..5)
# ════════════════════════════════════════════════════════════════
print("\n"+"="*60)
print("Generating Fourier bar-chart GIF — 4 subplots")
print("="*60)

_N_four  = 80    # time steps in the animation
_fps_fou = 6     # slow enough to read the values
_pause_f = 40    # freeze ~7 s at end
_t_four  = np.linspace(0, _T, _N_four)
_NLEV    = 5     # number of energy levels
_x_fou   = torch.linspace(0, 1, 500).view(-1,1).to(device)
_dx_fou  = 1.0 / (len(_x_fou)-1)

def _phi_n(n, x_np):
    """Exact eigenfunction  φ_n(x) = √2 sin(nπx)"""
    return np.sqrt(2)*np.sin(n*np.pi*x_np)

def _fourier_probs_nn(get_psi_fn, t_vals, x_tensor, dx):
    """Compute |c_n(t)|² for n=1..5 via numerical integration for a NN method."""
    x_np = x_tensor.cpu().numpy().flatten()
    P = np.zeros((len(t_vals), _NLEV))
    with torch.no_grad():
        for i, tv in enumerate(t_vals):
            tl  = torch.ones_like(x_tensor)*tv
            psi = get_psi_fn(x_tensor, tl)
            R   = psi[:,0].cpu().numpy().flatten()
            I   = psi[:,1].cpu().numpy().flatten()
            psi_c = R + 1j*I
            for n in range(1, _NLEV+1):
                cn = np.sum(psi_c * _phi_n(n, x_np)) * dx
                P[i, n-1] = abs(cn)**2
    return P

# GRAPE: coefficients come directly from the basis representation
# traj_g[k] is the coefficient vector in the energy eigenbasis
_gi_four = np.linspace(0, _Nt, _N_four, dtype=int)
_P_grape = np.zeros((_N_four, _NLEV))
for _i, _k in enumerate(_gi_four):
    for _n in range(_NLEV):
        _P_grape[_i, _n] = abs(traj_g[_k][_n])**2   # |c_n|² directly

# MLP, KAN, Hybrid: numerical integration
print("  Computing MLP Fourier coefficients …")
_P_mlp  = _fourier_probs_nn(mlp.get_psi,    _t_four, _x_fou, _dx_fou)
print("  Computing KAN Fourier coefficients …")
_P_kan  = _fourier_probs_nn(kan.get_psi,    _t_four, _x_fou, _dx_fou)
print("  Computing Hybrid Fourier coefficients …")
_P_hyb  = _fourier_probs_nn(hybrid.psi,     _t_four, _x_fou, _dx_fou)

def _normalise(P):
    """Divide each row so bars sum to 1."""
    s = P.sum(axis=1, keepdims=True)
    s[s < 1e-12] = 1.0   # avoid division by zero at t=0 edge cases
    return P / s

_Pn_grape = _normalise(_P_grape)
_Pn_mlp   = _normalise(_P_mlp)
_Pn_kan   = _normalise(_P_kan)
_Pn_hyb   = _normalise(_P_hyb)

_lev_labels  = [f"n={n}" for n in range(1, _NLEV+1)]
_bar_colors  = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']
_four_data   = [
    ("GRAPE",  _Pn_grape, "darkgreen"),
    ("MLP",    _Pn_mlp,   "purple"),
    ("KAN",    _Pn_kan,   "darkorange"),
    ("Hybrid", _Pn_hyb,   "darkorchid"),
]
_frames_f = list(range(_N_four)) + [_N_four-1]*_pause_f

fig_fou, axes_fou = plt.subplots(2, 2, figsize=(14, 10))
fig_fou.suptitle(
    r"Fourier probabilities $|c_n|^2$ (normalised, $\sum_n |c_n|^2 = 1$) — All Methods",
    fontsize=13, fontweight='bold')
fig_fou.patch.set_facecolor('white')

_bar_containers = []
_title_txts_f   = []

for idx, (name, Pn_, col_) in enumerate(_four_data):
    ax = axes_fou.flat[idx]; ax.set_facecolor('white')
    # Initial bar chart (frame 0)
    bars = ax.bar(_lev_labels, Pn_[0], color=_bar_colors,
                  edgecolor='black', linewidth=1.2)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(r"$|c_n|^2$ (normalised)", fontsize=10)
    ax.set_xlabel("Energy level", fontsize=10)
    ax.set_title(f"{name}  —  $t = 0.000$", fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.axhline(1.0/5, color='gray', ls=':', lw=1, alpha=0.5)   # uniform reference
    _bar_containers.append(bars)
    _title_txts_f.append(ax)

plt.tight_layout()

def _update_fou(frame):
    i = min(frame, _N_four-1)
    tc = _t_four[i]
    for idx, (name, Pn_, col_) in enumerate(_four_data):
        for bar, h in zip(_bar_containers[idx], Pn_[i]):
            bar.set_height(h)
        _title_txts_f[idx].set_title(
            f"{name}  —  $t = {tc:.3f}$", fontsize=12)
    return []

print("  Rendering Fourier GIF …")
_anim_fou = FuncAnimation(fig_fou, _update_fou, frames=_frames_f,
                          interval=int(1000/_fps_fou), blit=False)
_anim_fou.save(f"{OUT}/gif_fourier_bars_4methods.gif",
               writer=PillowWriter(fps=_fps_fou))
plt.close()
print("Saved gif_fourier_bars_4methods.gif")

# ════════════════════════════════════════════════════════════════
#  GIF C — f(t) growing curve — 4-subplot version (SLOW, same scale)
# ════════════════════════════════════════════════════════════════
print("\n"+"="*60)
print("Generating f(t) integral GIF — 4 subplots (slow)")
print("="*60)

_N_int   = 160      # animation frames
_fps_int = 5        # slow so the value is readable
_pause_i = 50       # freeze for 10 s at end
_frames_i = list(range(_N_int)) + [_N_int-1]*_pause_i

_int_rows = [
    ("GRAPE",  _tmid,   ri_g, "darkgreen"),
    ("MLP",    tp_m_np, ri_m, "purple"),
    ("KAN",    tp_k_np, ri_k, "darkorange"),
    ("Hybrid", tp_h_np, ri_h, "darkorchid"),
]

# Shared y-limit across all panels
_gif_fi_ymax = max(ri_g[-1], ri_m[-1], ri_k[-1], ri_h[-1]) * 1.12

fig_i4, axes_i4 = plt.subplots(2, 2, figsize=(14, 10))
fig_i4.suptitle(
    r"$f(t) = \int_0^t |u(s)|^2\,ds$ — Accumulated Control Energy — All Methods",
    fontsize=14, fontweight='bold')
fig_i4.patch.set_facecolor('white')

_lines4 = []
_dots4  = []
_txts4  = []
_axes4  = list(axes_i4.flat)

for idx, (name, t_, ri_, col_) in enumerate(_int_rows):
    ax = _axes4[idx]; ax.set_facecolor('white')
    # Faint final curve as reference
    ax.plot(t_, ri_, color=col_, lw=1.5, alpha=0.20)
    ax.set_xlim(t_[0], t_[-1]); ax.set_ylim(0, _gif_fi_ymax)
    ax.set_title(f"{name}  (total $f(T)={ri_[-1]:.3f}$)", fontsize=12)
    ax.set_xlabel("$t$", fontsize=11)
    ax.set_ylabel(r"$f(t) = \int_0^t |u|^2\,ds$", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    ln, = ax.plot([], [], color=col_, lw=2.5)
    dot, = ax.plot([], [], 'o', color=col_, ms=7, zorder=5)
    _lines4.append(ln); _dots4.append(dot)
    txt = ax.text(0.97, 0.12, '', transform=ax.transAxes,
                  ha='right', va='bottom', fontsize=12, fontweight='bold',
                  bbox=dict(fc='white', ec='gray', boxstyle='round,pad=0.35', alpha=0.95))
    _txts4.append(txt)

plt.tight_layout()

def _update_gif_fi4(frame):
    for idx, (name, t_, ri_, col_) in enumerate(_int_rows):
        k = min(int(frame * (len(t_)-1) / (_N_int-1)), len(t_)-1)
        _lines4[idx].set_data(t_[:k+1], ri_[:k+1])
        _dots4[idx].set_data([t_[k]], [ri_[k]])
        _txts4[idx].set_text(f"$t={t_[k]:.3f}$\n$f(t)={ri_[k]:.3f}$")
    return _lines4 + _dots4 + _txts4

print("  Rendering GIF …")
_anim_i4 = FuncAnimation(fig_i4, _update_gif_fi4, frames=_frames_i,
                          interval=int(1000/_fps_int), blit=True)
_anim_i4.save(f"{OUT}/gif_f_t_integral_4panels.gif",
              writer=PillowWriter(fps=_fps_int))
plt.close()
print("Saved gif_f_t_integral_4panels.gif")

# ════════════════════════════════════════════════════════════════
#  GIF C — f(t) growing curve — single figure, all 4 methods
# ════════════════════════════════════════════════════════════════
print("\n"+"="*60)
print("Generating f(t) integral GIF — single figure (slow)")
print("="*60)

fig_i1, ax_i1 = plt.subplots(figsize=(13, 6))
fig_i1.patch.set_facecolor('white'); ax_i1.set_facecolor('white')

# Faint reference: full f(t) curves
for name, t_, ri_, col_ in [(n,t,r,c) for n,t,_u2,r,c in
                              [("GRAPE",_tmid,u2_g,ri_g,"darkgreen"),
                               ("MLP",tp_m_np,u2_m,ri_m,"purple"),
                               ("KAN",tp_k_np,u2_k,ri_k,"darkorange"),
                               ("Hybrid",tp_h_np,u2_h,ri_h,"darkorchid")]]:
    ax_i1.plot(t_, ri_, color=col_, lw=1.5, alpha=0.18)

ax_i1.set_xlim(*_TIME_XLIM)
ax_i1.set_ylim(0, _gif_fi_ymax)
ax_i1.set_title(
    r"$f(t) = \int_0^t |u(s)|^2\,ds$ — Accumulated Control Energy — All Methods",
    fontsize=13)
ax_i1.set_xlabel("$t$", fontsize=12)
ax_i1.set_ylabel(r"$f(t) = \int_0^t |u|^2\,ds$", fontsize=12)
ax_i1.grid(True, linestyle='--', alpha=0.55)

_lines1 = []; _dots1 = []; _txts1 = []
_ypos   = [0.97, 0.82, 0.67, 0.52]   # stacked text box positions

_int_rows1 = [
    ("GRAPE",  _tmid,   ri_g, "darkgreen"),
    ("MLP",    tp_m_np, ri_m, "purple"),
    ("KAN",    tp_k_np, ri_k, "darkorange"),
    ("Hybrid", tp_h_np, ri_h, "darkorchid"),
]

for idx, (name, t_, ri_, col_) in enumerate(_int_rows1):
    ln,  = ax_i1.plot([], [], color=col_, lw=2.5,
                      label=f"{name}  $f(T)={ri_[-1]:.2f}$")
    dot, = ax_i1.plot([], [], 'o', color=col_, ms=7, zorder=5)
    _lines1.append(ln); _dots1.append(dot)
    txt = ax_i1.text(0.985, _ypos[idx], '', transform=ax_i1.transAxes,
                     ha='right', va='top', fontsize=11, fontweight='bold',
                     color=col_,
                     bbox=dict(fc='white', ec=col_, boxstyle='round,pad=0.3',
                               alpha=0.92, lw=1.2))
    _txts1.append(txt)

ax_i1.legend(loc='upper left', fontsize=10, framealpha=0.9)
plt.tight_layout()

def _update_gif_fi1(frame):
    for idx, (name, t_, ri_, col_) in enumerate(_int_rows1):
        k = min(int(frame * (len(t_)-1) / (_N_int-1)), len(t_)-1)
        _lines1[idx].set_data(t_[:k+1], ri_[:k+1])
        _dots1[idx].set_data([t_[k]], [ri_[k]])
        _txts1[idx].set_text(f"{name}: $t={t_[k]:.3f}$  $f={ri_[k]:.3f}$")
    return _lines1 + _dots1 + _txts1

print("  Rendering GIF …")
_anim_i1 = FuncAnimation(fig_i1, _update_gif_fi1, frames=_frames_i,
                          interval=int(1000/_fps_int), blit=True)
_anim_i1.save(f"{OUT}/gif_f_t_integral_single.gif",
              writer=PillowWriter(fps=_fps_int))
plt.close()
print("Saved gif_f_t_integral_single.gif")

# ════════════════════════════════════════════════════════════════
#  SUMMARY
# ════════════════════════════════════════════════════════════════
print("\n"+"="*60)
print(f"All outputs saved to: {OUT}")
print("="*60)
_files = [
    "fig01_initial_states_2x2.png",
    "fig02_final_states_2x2.png",
    "fig03_states_all_methods.png",
    "fig04_norm_all_in_one.png",
    "fig05_norm_subplots_2x2.png",
    "fig06_u_subplots_2x2.png",
    "fig07_u2_subplots_2x2.png",
    "fig08_u2_all_in_one.png",
    "fig09_f_t_all_methods.png",
    "fig10_f_t_subplots_2x2.png",
    "gif_norm_psi_4methods.gif",
    "gif_wave_evolution_4methods.gif",
    "gif_fourier_bars_4methods.gif",
    "gif_f_t_integral_4panels.gif",
    "gif_f_t_integral_single.gif",
]
for f in _files:
    print(f"  {OUT}/{f}")

print(f"\nGRAPE   Fidelity      : {F_g:.7f}")
print(f"MLP     Control Energy: {E_m.item():.3f}")
print(f"KAN     Control Energy: {E_k.item():.3f}")
print(f"Hybrid  Control Energy: {E_h.item():.3f}")
print("="*60)

if IPYTHON_AVAILABLE:
    for f in ["gif_norm_psi_4methods.gif",
              "gif_wave_evolution_4methods.gif",
              "gif_fourier_bars_4methods.gif",
              "gif_f_t_integral_4panels.gif",
              "gif_f_t_integral_single.gif"]:
        display(Image(filename=f"{OUT}/{f}"))
