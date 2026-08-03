import io, csv, itertools, time
import copy
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from style import load_css, chip, style_matplotlib


load_css()
style_matplotlib()   # add this line

# ─────────────────────────── Helper: tau sampling ───────────────────────────
def sample_tau(n, early_frac, rng):   # early_frac of points forced into tau<0.2 (the leverage window)
    n_e = int(round(early_frac * n)); n_l = n - n_e; pts = []
    if n_e > 0:
        eb = np.linspace(0.0, 0.2, n_e + 1); pts += [rng.uniform(eb[k], eb[k+1]) for k in range(n_e)]
    if n_l > 0:
        lb = np.linspace(0.0, 1.0, n_l + 1); pts += [rng.uniform(lb[k], lb[k+1]) for k in range(n_l)]
    return np.array(pts)

def early_biased_times(n_meas, early_frac=0.5, early_end=0.2):
    """Stratified measurement times, biased toward early time.

    Puts `early_frac` of the measurements (stratified) in [0, early_end] and the
    remainder (stratified) in [early_end, 1]. This concentrates measurement density
    in tau < 0.2, where the two-run viscosity leverage c(tau) is concentrated --
    the region that actually makes Psi identifiable.

    Uses the global np.random, so your existing np.random.seed(seed) still makes
    it reproducible.
    """
    n_early = int(round(early_frac * n_meas))
    n_late = n_meas - n_early
    t = []
    if n_early > 0:
        eb = np.linspace(0.0, early_end, n_early + 1)
        t += [float(np.random.uniform(eb[i], eb[i + 1])) for i in range(n_early)]
    if n_late > 0:
        lb = np.linspace(early_end, 1.0, n_late + 1)
        t += [float(np.random.uniform(lb[i], lb[i + 1])) for i in range(n_late)]
    return np.sort(np.asarray(t))


def verdict(ms, hor, fit_errs=None, Ld_A=None, Ld_C=None):
    def _num(x): return isinstance(x, float) and x == x
    v = {}
    if ms is not None and len(ms) > 0:
        A = np.array([a for a, _ in ms]); d = np.array([b for _, b in ms])
        arel = float(np.std(A) / max(abs(np.median(A)), 1e-6))
        drel = float(np.std(d) / max(abs(np.median(d)), 1e-6))
    else:
        arel = drel = float('nan')

    worst = float(max(fit_errs)) if fit_errs else None
    if worst is None:
        v['h'] = 'HIGH — fit to data by construction'; h_bad = False
    elif worst < 15:
        v['h'] = 'HIGH — h tracks the data (worst run %.1f%%)' % worst; h_bad = False
    elif worst < 40:
        v['h'] = 'MEDIUM — physics bent h off the data (worst run %.1f%%)' % worst; h_bad = True
    else:
        v['h'] = ('LOW — physics PULLED h OFF the data (worst run %.1f%%): '
                  'the ODE cannot reproduce these traces' % worst); h_bad = True

    if Ld_A is not None and Ld_C is not None and Ld_A > 1e-12:
        ratio = Ld_C / Ld_A
        if ratio > 5:
            v['consistency'] = ('FLAG — joint data-loss %.1fx the data-only fit: '
                                'physics and data disagree' % ratio)
        elif ratio > 2:
            v['consistency'] = ('WARN — joint data-loss %.1fx the data-only fit: '
                                'physics strains the data' % ratio)
        else:
            v['consistency'] = 'OK — data consistent with the ODE (joint %.1fx data-only)' % ratio
    else:
        v['consistency'] = 'consistency not measured (retrain to enable)'

    v['E'] = 'LOW — derived from a mis-fit h' if h_bad else 'MEDIUM-HIGH — slope of h'
    v['combined'] = ('COMPROMISE — physics & data disagree, split is suspect' if h_bad
                     else ('MEDIUM — combined only loosely pinned'
                           if (worst is not None and worst >= 15)
                           else 'HIGH — what the physics loss pins'))
    v['Psi amplitude'] = ('MEDIUM (reproducible across restarts)' if (_num(arel) and arel < 0.3)
                          else ('LOW (multi-start spread large)' if _num(arel) else 'n/a'))
    v['Psi decay'] = ('UNIDENTIFIABLE — a prior-driven extrapolation, NOT a measurement'
                      if (_num(drel) and drel > 0.5)
                      else ('LOW-MEDIUM (treat cautiously)' if _num(drel) else 'n/a'))

    note = (("Information horizon: %.0f%% of the 2-run Psi leverage sits in tau<0.2."
             % (100 * hor['frac_early'])) if hor
            else "Single run -> no multi-run lever on Psi.")

    bad  = h_bad or (Ld_A is not None and Ld_C is not None and Ld_A > 1e-12 and Ld_C / Ld_A > 5)
    warn = (not bad) and (h_bad or (_num(drel) and drel > 0.5) or
                          (Ld_A is not None and Ld_C is not None and Ld_A > 1e-12 and Ld_C / Ld_A > 2))
    level = 'bad' if bad else ('warn' if warn else 'ok')
    return v, arel, drel, note, level


def status_card(level, title, lines):
    """Render a simple status card using Streamlit's native components."""
    colors = {'ok': '#22c55e', 'warn': '#f59e0b', 'bad': '#ef4444'}
    icons = {'ok': '✅', 'warn': '⚠️', 'bad': '🚨'}
    color = colors.get(level, '#64748b')
    icon = icons.get(level, '•')
    
    html = f'''
    <div style="border-left: 4px solid {color}; padding: 12px 16px; margin: 10px 0; 
                background: #f8fafc; border-radius: 4px;">
        <div style="font-weight: 600; font-size: 1.1em; margin-bottom: 8px;">{icon} {title}</div>
        <div style="font-family: monospace; font-size: 0.9em; line-height: 1.6;">
            {'<br>'.join(str(line) for line in lines)}
        </div>
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)


def consistency_probe(param, runs, steps=300, lr=1e-3):
    """Post-hoc consistency probe. Compares the data-loss of the JOINT-trained h_nets
    against a short DATA-ONLY refit of the same h_nets (psi/en frozen, copy discarded).
    If a data-only refit drops the data loss a lot, the joint solution was being pulled
    OFF the data by the physics term -> the runs are inconsistent with the ODE.
    Returns ratio = joint_loss / refit_loss (>=1), or None on failure."""
    import torch.optim as optim
    hn = param['hn']

    def dloss(nets):
        tot = 0.0
        for i, r in enumerate(runs):
            tot += torch.mean((nets[i](r['td'], 1.0) - r['hd']) ** 2).item()
        return tot / len(runs)

    with torch.no_grad():
        joint = dloss(hn)
    refit = [copy.deepcopy(h) for h in hn]
    o = optim.Adam([p for h in refit for p in h.parameters()], lr=lr)
    for _ in range(steps):
        o.zero_grad()
        L = sum(torch.mean((refit[i](runs[i]['td'], 1.0) - runs[i]['hd']) ** 2)
                for i in range(len(runs)))
        L.backward(); o.step()
    with torch.no_grad():
        refit_loss = dloss(refit)
    if refit_loss <= 1e-10:
        return None
    return joint / refit_loss


# ─────────────────────────── Page & theme ───────────────────────────

st.set_page_config(page_title="SpinCoat PINN Lab", page_icon="🧪", layout="wide")
load_css()

st.caption("PHYSICS-INFORMED NEURAL NETWORK · INVERSE DISCOVERY")
st.markdown("## SpinCoat **:blue[PINN]** Lab")

st.markdown(
    chip("dh/dτ = −K(τ)·h³ − E(τ)")
    + chip("2 spin runs · shared Ψ & E")
    + chip("synthetic or your data", accent=True),
    unsafe_allow_html=True,
)

# ─────────────────────────── Model (unchanged base) ───────────────────────────
class ThicknessNet(nn.Module):
    def __init__(self, h=32, L=3):
        super().__init__()
        L_ = [nn.Linear(1, h), nn.Tanh()]
        for _ in range(L - 1): L_ += [nn.Linear(h, h), nn.Tanh()]
        L_ += [nn.Linear(h, 1)]
        self.net = nn.Sequential(*L_); self.sp = nn.Softplus()
    def forward(self, tau, h0=1.0):
        return h0 - tau * self.sp(self.net(tau))   # hard-enforces h(0)=h0

class PsiNet(nn.Module):
    def __init__(self, h=32, L=3):
        super().__init__()
        L_ = [nn.Linear(1, h), nn.Tanh()]
        for _ in range(L - 1): L_ += [nn.Linear(h, h), nn.Tanh()]
        L_ += [nn.Linear(h, 1)]
        self.net = nn.Sequential(*L_)
    def forward(self, tau): return torch.exp(self.net(tau))

class PsiPar(nn.Module):   # constrained Psi = A*exp(-d*tau), d>=0 (paper §4.4)
    def __init__(self):
        super().__init__()
        self.logA = nn.Parameter(torch.tensor(0.0))
        self.raw  = nn.Parameter(torch.tensor(0.5))
        self.sp   = nn.Softplus()
    def forward(self, t):               # same call signature as PsiNet -> residual() unchanged
        return torch.exp(self.logA - self.sp(self.raw) * t)

class ETildeNet(nn.Module):
    def __init__(self, h=32, L=3):
        super().__init__()
        L_ = [nn.Linear(1, h), nn.Tanh()]
        for _ in range(L - 1): L_ += [nn.Linear(h, h), nn.Tanh()]
        L_ += [nn.Linear(h, 1)]
        self.net = nn.Sequential(*L_); self.sp = nn.Softplus()
    def forward(self, tau): return self.sp(self.net(tau))

# ================= E CORRECTION: fixed-Ẽ modes =================
class FixedE(nn.Module):
    """Constant evaporation Ẽ(τ)=E_B. Zero trainable params -> optimizer cannot touch it."""
    def __init__(self, E_B):
        super().__init__()
        self.register_buffer("E_B", torch.tensor(float(E_B), dtype=torch.float32))
    def forward(self, tau):
        return self.E_B.expand_as(tau)

class FixedEExp(nn.Module):
    """Parametric evaporation Ẽ(τ)=E_B·exp(−E_d·τ). Zero trainable params."""
    def __init__(self, E_B, E_d):
        super().__init__()
        self.register_buffer("E_B", torch.tensor(float(E_B), dtype=torch.float32))
        self.register_buffer("E_d", torch.tensor(float(E_d), dtype=torch.float32))
    def forward(self, tau):
        return self.E_B * torch.exp(-self.E_d * tau)

# ═══════════════════ FIXES LAB HELPERS ═══════════════════
def inv_softplus(x):
    x = float(np.clip(x, 1e-3, 20.0)); return float(np.log(np.expm1(x)))

class PsiParNet(nn.Module):                      # Fix: parametrized Ψ = A·e^(−βτ)
    def __init__(self):
        super().__init__()
        self.lA = nn.Parameter(torch.tensor(0.0)); self.lB = nn.Parameter(torch.tensor(0.0))
        self.sp = nn.Softplus()
    def forward(self, tau):
        return (self.sp(self.lA)+1e-3) * torch.exp(-(self.sp(self.lB)+1e-3)*tau)

class EConstNet(nn.Module):                      # Fix: E = const
    def __init__(self, e0=1.0):
        super().__init__(); self.p = nn.Parameter(torch.tensor(inv_softplus(e0))); self.sp = nn.Softplus()
    def forward(self, tau): return self.sp(self.p) * torch.ones_like(tau)

class EExpNet(nn.Module):                        # Fix: E = E_B·e^(−E_d·τ)
    def __init__(self, e0=1.0, d0=1.0):
        super().__init__()
        self.pB = nn.Parameter(torch.tensor(inv_softplus(e0)))
        self.pD = nn.Parameter(torch.tensor(inv_softplus(d0))); self.sp = nn.Softplus()
    def forward(self, tau): return self.sp(self.pB) * torch.exp(-self.sp(self.pD)*tau)

def estimate_E_late_sweep(data, tau_cut=0.5):          # Fix: autofill E_B,E_d from late-time slope
    EB, ED = [], []
    for r in data["runs"]:
        o = np.argsort(r["tau_s"]); t, h = r["tau_s"][o], r["h_meas"][o]
        m = t >= tau_cut; t, h = t[m], h[m]
        if len(t) < 3: continue
        dt = np.diff(t); ok = dt > 1e-8
        tm = 0.5*(t[1:]+t[:-1])[ok]; s = (-np.diff(h)/dt)[ok]
        pos = s > 1e-8
        if pos.sum() < 2: continue
        c = np.polyfit(tm[pos], np.log(s[pos]), 1)
        ED.append(-c[0]); EB.append(np.exp(c[1]))
    if not EB: return None
    return (float(np.clip(np.median(EB),1e-3,1e2)), float(np.clip(np.median(ED),0.0,10.0)))

def estimate_E_late(runs, k=4):
    """Anchor Ẽ from the late-time slope, where ĥ³≈0 so dh̃/dτ ≈ −Ẽ.
    Pools the last k points of every run, log-linear fit of −slope vs τ.
    Returns (E_B, E_d)."""
    ts, ys = [], []
    for r in runs:
        t = np.asarray(r["tau_s"], float); h = np.asarray(r["h_meas"], float)
        if len(t) < max(3, k): continue
        t, h = t[-k:], h[-k:]
        sl = np.gradient(h, t)
        m = sl < 0
        if m.sum() < 2: continue
        ts.append(t[m]); ys.append(np.log(-sl[m]))
    if not ts: return 3.0, 0.0
    slope, intercept = np.polyfit(np.concatenate(ts), np.concatenate(ys), 1)
    return float(np.exp(intercept)), float(-slope)

def residual(h_net, psi_net, e_net, tau, w_norm):
    h = h_net(tau, h0=1.0)
    dh = torch.autograd.grad(h, tau, torch.ones_like(h), create_graph=True, retain_graph=True)[0]
    K = (w_norm ** 2) * psi_net(tau)
    return dh + K * h**3 + e_net(tau), h, K

# ─────────────────────────── Physics / data (unchanged base) ───────────────────────────
def simulate(psi_A, psi_d, E_B, E_d, w, tau):
    def rhs(t, h): return [-(w**2) * psi_A*np.exp(-psi_d*t) * h[0]**3 - E_B*np.exp(-E_d*t)]
    return solve_ivp(rhs, (0, 1), [1.0], t_eval=tau, method="RK45").y[0]

@st.cache_data
# ─────────────────────────── Dense-early sampling helper ───────────────────────────
def _sample_tau(n_meas, dense_early, early_frac, early_span, seed):
    """Stratified sample times; optionally concentrate a fraction of the points
    in the early window [0, early_span] where the viscosity signal lives."""
    rng = np.random.default_rng(seed)
    if not dense_early:                                   # unchanged default
        edges = np.linspace(0.0, 1.0, n_meas + 1)
        return np.array([rng.uniform(edges[i], edges[i + 1]) for i in range(n_meas)])
    n_e = max(1, int(round(early_frac * n_meas)))
    n_l = max(1, n_meas - n_e)
    pts = []
    ee = np.linspace(0.0, early_span, n_e + 1)            # dense early block
    pts += [rng.uniform(ee[i], ee[i + 1]) for i in range(n_e)]
    le = np.linspace(early_span, 1.0, n_l + 1)            # remainder covers the tail
    pts += [rng.uniform(le[i], le[i + 1]) for i in range(n_l)]
    return np.array(pts)

@st.cache_data
def generate_data(psi_A, psi_d, E_B, E_d, rpm_a, rpm_b, n_meas, noise, n_colloc, seed,
                  dense_early=False, early_frac=0.5, early_span=0.2):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    tau = np.linspace(0, 1, 500); w_ref = rpm_a
    runs, K_true = [], []
    for rpm in (rpm_a, rpm_b):
        w = rpm / w_ref
        h = simulate(psi_A, psi_d, E_B, E_d, w, tau)
        K_true.append((w**2) * psi_A*np.exp(-psi_d*tau))
        tgt = _sample_tau(n_meas, dense_early, early_frac, early_span, seed)
        idx = np.sort(np.unique([np.argmin(np.abs(tau - t)) for t in tgt]))
        h_s = h[idx]; meas = np.clip(h_s + rng.normal(0, noise, len(idx)) * h_s, 1e-4, None)
        runs.append(dict(rpm=rpm, w=w, h=h, tau_s=tau[idx], h_meas=meas,
                         tau_c=np.sort(rng.uniform(0, 1, n_colloc))))
    return dict(tau=tau, runs=runs, K_true=K_true,
                psi=psi_A*np.exp(-psi_d*tau), e=E_B*np.exp(-E_d*tau), has_truth=True)

# ─────────────────────────── MANUAL / CSV helpers (ADDED) ───────────────────────────
_MANUAL_EXAMPLE = ("run_id,t,h,rpm\n"
                   "0,0,1.000,3000\n0,10,0.430,3000\n0,20,0.180,3000\n0,30,0.084,3000\n"
                   "1,0,1.000,4500\n1,10,0.250,4500\n1,20,0.070,4500\n1,30,0.012,4500")

def _parse_rows(text):
    """Return list of (run_id, t, h, rpm|None, h_true|None). Header needs t & h."""
    out = []
    rd = csv.DictReader(io.StringIO(text))
    if not rd.fieldnames:
        return out
    for raw in rd:
        row = {k.strip().lower().lstrip("﻿"): (v.strip() if isinstance(v, str) else v)
               for k, v in raw.items()}
        try:
            t = float(row["t"]); h = float(row["h"])
        except (KeyError, ValueError, TypeError):
            continue
        rid_s = row.get("run_id", "")
        rid = int(float(rid_s)) if rid_s not in ("", None) else 0
        rpm_s = row.get("rpm", "")
        rpm = float(rpm_s) if rpm_s not in ("", None) else None
        ht_s = row.get("h_true", "")
        ht = float(ht_s) if ht_s not in ("", None) else None
        out.append((rid, t, h, rpm, ht))
    return out

def _build_manual(rows, h_wet, t_ref, default_rpm):
    """Group parsed rows into the same dict-shape generate_data uses (no truth)."""
    if h_wet <= 0 or t_ref <= 0:
        return None
    g = {}
    for rid, t, h, rpm, ht in rows:
        g.setdefault(rid, dict(t=[], h=[], rpm=rpm, ht=[]))
        g[rid]["t"].append(t); g[rid]["h"].append(h)
        if rpm is not None: g[rid]["rpm"] = rpm
        if ht is not None: g[rid]["ht"].append(ht)
    rps = [g[k]["rpm"] for k in g if g[k]["rpm"] is not None]
    rpm_ref = float(min(rps)) if rps else float(default_rpm)
    rng = np.random.default_rng(7)
    runs = []
    for rid in sorted(g):
        if len(g[rid]["t"]) < 2:
            continue
        o = np.argsort(g[rid]["t"])
        t_raw = np.array(g[rid]["t"])[o]; h_raw = np.array(g[rid]["h"])[o]
        ht_raw = np.array(g[rid]["ht"])[o] if g[rid]["ht"] else None
        rpm = g[rid]["rpm"] if g[rid]["rpm"] is not None else float(default_rpm)
        w = rpm / rpm_ref if rpm_ref > 0 else 1.0
        runs.append(dict(
            rpm=rpm, w=w,
            tau_s=t_raw / float(t_ref),
            h_meas=h_raw / float(h_wet),
            tau_c=np.sort(rng.uniform(0, 1, 200)),
            h=(ht_raw / float(h_wet)) if ht_raw is not None else None))
    if not runs:
        return None
    return dict(tau=np.linspace(0, 1, 300), runs=runs, has_truth=False,
                psi=None, e=None, K_true=None,
                manual_meta=dict(h_wet=float(h_wet), t_ref=float(t_ref), rpm_ref=rpm_ref))

# ─────────────────────────── Training / eval (base; .index -> enumerate) ───────────────────────────
def train(data, h, L, epochs, lr, w_d, w_p, seed, prog, ph, param_psi=False, mono_w=0.0, reweight_h3=False, fix_E=None):
    torch.manual_seed(seed)
    h_nets = [ThicknessNet(h, L) for _ in data["runs"]]
    psi = PsiPar() if param_psi else PsiNet(h, L)
    e = fix_E if fix_E is not None else ETildeNet(h, L)
    
    # Phase A: data-only training
    pA = [p for n in h_nets for p in n.parameters()]
    oA = optim.Adam(pA, lr=lr)
    Ld_A = 0.0
    for ep in range(epochs // 2):
        oA.zero_grad(); Ld = 0.0
        for i, r in enumerate(data["runs"]):
            td = torch.tensor(r["tau_s"], dtype=torch.float32).reshape(-1, 1)
            hd = torch.tensor(r["h_meas"], dtype=torch.float32).reshape(-1, 1)
            Ld = Ld + torch.mean((h_nets[i](td, 1.0) - hd) ** 2)
        nrun = len(data["runs"])
        Ld = Ld / nrun
        Ld.backward(); oA.step()
        Ld_A = float(Ld.item())
        if ep % 20 == 0 or ep == (epochs // 2) - 1:
            prog.progress((ep + 1) / (epochs // 2) * 0.5)
            ph.caption(f"Phase A epoch {ep+1}/{epochs//2} · L_data {Ld_A:.5f}")
    
    # Phase C: joint training (data + physics)
    pC = [p for n in h_nets for p in n.parameters()] + list(psi.parameters()) + list(e.parameters())
    oC = optim.Adam(pC, lr=lr)
    hist = dict(d=[], p=[], t=[])
    Ld_C = 0.0
    for ep in range(epochs // 2, epochs):
        oC.zero_grad(); Ld = Lp = Lmono = 0.0
        for i, r in enumerate(data["runs"]):
            td = torch.tensor(r["tau_s"], dtype=torch.float32).reshape(-1, 1)
            hd = torch.tensor(r["h_meas"], dtype=torch.float32).reshape(-1, 1)
            Ld = Ld + torch.mean((h_nets[i](td, 1.0) - hd) ** 2)
            tc = torch.tensor(r["tau_c"], dtype=torch.float32).reshape(-1, 1).requires_grad_(True)
            res, hh, _ = residual(h_nets[i], psi, e, tc, r["w"])
            if reweight_h3:
                wgt = 1.0 / (hh.detach() ** 3 + 1e-3)
                wgt = wgt / wgt.mean()
                Lp = Lp + torch.mean((wgt * res) ** 2)
            else:
                Lp = Lp + torch.mean(res ** 2)
            if mono_w > 0.0:
                pt = psi(tc); dpt = torch.autograd.grad(pt.sum(), tc, create_graph=True, retain_graph=True)[0]
                Lmono = Lmono + torch.mean(torch.relu(dpt)) ** 2
        nrun = len(data["runs"])
        Ld = Ld / nrun; Lp = Lp / nrun
        if mono_w > 0.0:
            Lmono = Lmono / nrun
        loss = w_d * Ld + w_p * Lp + mono_w * Lmono; loss.backward(); oC.step()
        hist["d"].append(Ld.item()); hist["p"].append(Lp.item()); hist["t"].append(loss.item())
        Ld_C = float(Ld.item())
        if ep % 20 == 0 or ep == epochs - 1:
            prog.progress(0.5 + (ep - epochs // 2 + 1) / (epochs - epochs // 2) * 0.5)
            ph.caption(f"Phase C epoch {ep+1}/{epochs} · L_data {Ld_C:.5f} · L_phys {Lp.item():.5f}")

    return dict(hn=h_nets, psi=psi, en=e, Ld_A=Ld_A, Ld_C=Ld_C), hist

def evaluate(nets, data):
    with torch.no_grad():
        t = torch.tensor(data["tau"], dtype=torch.float32).reshape(-1, 1)
        psi, e = nets["psi"](t).numpy().flatten(), nets["en"](t).numpy().flatten()
        hs = [n(t, 1.0).numpy().flatten() for n in nets["hn"]]
    return dict(psi=psi, e=e, hs=hs, Ks=[(r["w"]**2)*psi for r in data["runs"]])

def algebraic_split(h_nets, w0, w1, tau_d):
    """Hybrid recovery: PINN denoises h; the Psi/E split is solved directly.
    At each tau, the two runs give  w_i^2 h_i^3 Psi + E = -dh_i/dtau,
    a 2x2 system in (Psi, E). Determinant D = w0^2 h0^3 - w1^2 h1^3 is the
    two-run leverage: where |D|->0 the split is undefined (masked to NaN),
    so the returned curve literally ends where the information ends."""
    tau_d = np.asarray(tau_d, dtype=float)
    h, dh = [], []
    for net in h_nets[:2]:
        tg = torch.tensor(tau_d, dtype=torch.float32).reshape(-1, 1).requires_grad_(True)
        hv = net(tg, 1.0)
        ghv = torch.autograd.grad(hv, tg, grad_outputs=torch.ones_like(hv),
                                  create_graph=False)[0]
        h.append(hv.detach().numpy().ravel())
        dh.append(ghv.detach().numpy().ravel())
    h0, h1 = h; g0, g1 = dh
    D = (w0 ** 2) * h0 ** 3 - (w1 ** 2) * h1 ** 3
    thr = 0.05 * abs(float(D[0])) if len(D) and abs(float(D[0])) > 0 else 1e-6
    Psi = (g1 - g0) / D
    E = (-(w0 ** 2) * h0 ** 3 * g1 + (w1 ** 2) * h1 ** 3 * g0) / D
    mask = (np.abs(D) > thr) & (np.abs(Psi) < 10) & (np.abs(E) < 10)   # drop horizon spikes
    Psi = np.where(mask, Psi, np.nan); E = np.where(mask, E, np.nan)
    return dict(tau=tau_d, Psi=Psi, E=E, D=D, thr=thr)

# ─────────────────────────── Plot helper (base) ───────────────────────────
plt.rcParams.update({"figure.facecolor": "none", "axes.facecolor": "none",
                     "axes.edgecolor": "#4fc3f7", "axes.labelcolor": "#e0e0e0",
                     "text.color": "#e0e0e0", "axes.grid": True, "grid.color": "#2d3748",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "Calibri"})
CY, AM, RD = "#4fc3f7", "#ffb74d", "#ef5350"
def ax0(a): a.tick_params(colors="#b0bec5"); a.grid(alpha=.3); return a

# ─────────────────────────── Sidebar ───────────────────────────
st.sidebar.markdown("### Controls")
source = st.sidebar.radio("Data source", ["Synthetic", "Manual / CSV"], index=0,
                          help="Synthetic = the app generates the hidden truth. "
                               "Manual / CSV = your own thickness-vs-time data.")
SRC_SYN = (source == "Synthetic")

with st.sidebar.expander("Physics", expanded=True):
    psi_A = st.slider("Ψ_A · convective strength", 0.1, 3.0, 1.2, 0.05)
    psi_d = st.slider("Ψ decay", 0.5, 6.0, 3.0, 0.1)
    E_B   = st.slider("E_B · evaporation strength", 0.5, 6.0, 3.0, 0.1)
    E_d   = st.slider("E decay", 0.5, 6.0, 3.5, 0.1)
    rpm_a = st.slider("Run A · RPM", 1000, 6000, 3000, 100)
    rpm_b = st.slider("Run B · RPM", 1000, 6000, 4500, 100)
with st.sidebar.expander("Synthetic data", expanded=SRC_SYN):
    n_meas  = st.slider("Measurements / run", 4, 24, 8)
    noise   = st.slider("Noise σ", 0.0, 0.10, 0.02, 0.005)
    n_colloc = st.slider("Collocation points", 50, 400, 200, 10)
    seed = st.number_input("Seed", 0, 999, 42)
    dense_early = st.checkbox("Dense early sampling (viscosity window)", value=False,
                              help="Concentrate a chosen fraction of the measurements in the "
                                   "early window where the viscosity signal lives.")
    early_frac = st.slider("…fraction of points in early window", 0.2, 0.8, 0.5, 0.05,
                           disabled=not dense_early)
    early_span = st.slider("…early window span (τ)", 0.1, 0.4, 0.2, 0.05,
                           disabled=not dense_early)
with st.sidebar.expander("Training", expanded=True):
    epochs = st.slider("Epochs", 200, 30000, 1500, 100)
    lr = st.select_slider("Learning rate", [5e-4, 1e-3, 2e-3, 5e-3], value=1e-3)
    hid = st.slider("Hidden width", 16, 64, 32, 8)
    lay = st.slider("Hidden layers", 2, 5, 3)
    w_d = st.slider("W_data", 0.1, 5.0, 1.0, 0.1)
    w_p = st.slider("W_physics", 0.1, 5.0, 1.0, 0.1)
    param_psi = st.checkbox("Use parameterized Ψ (PsiPar)", value=False)
    mono_w = st.slider("Ψ monotonicity weight (enforce decay)", 0.0, 1.0, 0.0, 0.05)
    reweight_h3 = st.checkbox("Re-weight physics loss by 1/h³", value=False)

with st.sidebar.expander("Ẽ fixing (E correction)"):
    e_mode = st.radio("Ẽ mode", ["Free (learned)", "Fixed: constant", "Fixed: exponential"], index=0)
    if e_mode != "Free (learned)":
        if st.button("Auto-fill E_B, E_d from late-time slope"):
            _runs = (st.session_state.get("data") or {}).get("runs") \
                 or (st.session_state.get("manual") or {}).get("runs")
            if _runs:
                st.session_state["E_B_fix"], st.session_state["E_d_fix"] = estimate_E_late(_runs)
        E_B_fix = st.number_input("E_B (fixed)", 0.0, 10.0, 3.0, 0.1, key="E_B_fix")
        E_d_fix = st.number_input("E_d (fixed)", 0.0, 10.0, 3.5, 0.1, key="E_d_fix")
    fix_E = None
    if e_mode == "Fixed: constant":    fix_E = FixedE(E_B_fix)
    if e_mode == "Fixed: exponential": fix_E = FixedEExp(E_B_fix, E_d_fix)

st.sidebar.markdown("---")
if SRC_SYN:
    gen_btn = st.sidebar.button("Generate data", use_container_width=True, key="gen_btn")
else:
    st.sidebar.caption("Load your thickness data in the **Manual / CSV** tab, then Train.")
    gen_btn = False
train_btn = st.sidebar.button("Train PINN", use_container_width=True, key="train_btn")

for k in ("data", "nets", "hist"):
    st.session_state.setdefault(k, None)
if gen_btn:
    st.session_state.data = generate_data(psi_A, psi_d, E_B, E_d, rpm_a, rpm_b, n_meas, noise, n_colloc, seed,
                                          dense_early=dense_early, early_frac=early_frac, early_span=early_span)
    st.session_state.nets = st.session_state.hist = None

# ─────────────────────────── Tabs ───────────────────────────
tb = st.tabs(["Physics", "Data", "Train", "Results", "Manual / CSV", "Model", "Fixes Lab"])

# ---------- 0 · PHYSICS ----------
with tb[0]:
    st.markdown("#### Live thinning simulator")
    if not SRC_SYN:
        st.caption("This simulator draws the *synthetic* physics from the sliders, for intuition. "
                   "Your loaded data drives the Data / Train / Results tabs.")
    tau = np.linspace(0, 1, 500); w_ref = rpm_a
    c1, c2 = st.columns(2)
    with c1:
        f, a = plt.subplots(figsize=(6, 3.6), facecolor="none")
        for rpm, c in ((rpm_a, CY), (rpm_b, AM)):
            a.plot(tau, simulate(psi_A, psi_d, E_B, E_d, rpm/w_ref, tau), color=c, lw=2.4, label=f"{rpm} RPM")
        a.set_xlabel("τ"); a.set_ylabel("h(τ)"); a.set_title("Film thinning h(τ)")
        a.legend(frameon=False); ax0(a); st.pyplot(f)
    with c2:
        f, a = plt.subplots(figsize=(6, 3.6), facecolor="none")
        a.plot(tau, psi_A*np.exp(-psi_d*tau), color=CY, lw=2.4, label="Ψ(τ)")
        a.plot(tau, E_B*np.exp(-E_d*tau), color=AM, lw=2.4, label="E(τ)")
        a.set_xlabel("τ"); a.set_title("Latent Ψ(τ) & evaporation E(τ)")
        a.legend(frameon=False); ax0(a); st.pyplot(f)
    st.caption("K(τ) = (ω/ω_ref)²·Ψ(τ) — higher spin → stronger convective thinning.")

# ---------- 1 · DATA (base; handles manual with no true curve) ----------
with tb[1]:
    st.markdown("#### " + ("Synthetic sparse measurements" if SRC_SYN else "Loaded measurements"))
    if st.session_state.data:
        d = st.session_state.data
        truth = bool(d.get("has_truth", False))
        n = len(d["runs"])
        c1, c2 = st.columns(2)
        for i in range(min(n, 2)):
            with (c1 if i == 0 else c2):
                f, a = plt.subplots(figsize=(6, 3.6), facecolor="none")
                run = d["runs"][i]
                if truth and run.get("h") is not None:
                    a.plot(d["tau"], run["h"], color=[CY, AM][i % 2], lw=2.2, alpha=.5, label="true h(τ)")
                elif run.get("h") is not None:
                    a.plot(run["tau_s"], run["h"], color=[CY, AM][i % 2], lw=2.2, alpha=.5, label="true h @data")
                a.scatter(run["tau_s"], run["h_meas"], color=[CY, AM][i % 2], s=46, zorder=5, label="data")
                a.scatter(run["tau_c"], np.zeros_like(run["tau_c"]),
                          marker="|", color="#64748b", s=60, label="collocation")
                a.set_xlabel("τ"); a.set_ylabel("h"); a.set_title(f"Run {i} · {run.get('rpm', '?')} RPM")
                a.legend(frameon=False, fontsize=8); ax0(a); st.pyplot(f)
        if n > 2:
            st.caption(f"(showing first 2 of {n} runs)")
        if SRC_SYN:
            st.caption(f"{n_meas} stratified samples/run · σ={noise} · {n_colloc} collocation points · seed {seed}")
        else:
            m = d.get("manual_meta", {})
            st.caption(f"manual · {n} run(s) · h_wet={m.get('h_wet')} · t_ref={m.get('t_ref')} · rpm_ref={m.get('rpm_ref')}")
    else:
        st.info("Hit **Generate data** (synthetic) or load data in the **Manual / CSV** tab.")

# ---------- 2 · TRAIN (base; safe guard instead of silent synthetic) ----------
with tb[2]:
    st.markdown("#### Training")
    if train_btn:
        if st.session_state.data is None:
            st.warning("No data loaded yet. Use **Generate data** (synthetic) or load CSV / manual "
                       "data in the **Manual / CSV** tab first.")
            st.stop()
        prog = st.progress(0); ph = st.empty()
        st.session_state.nets, st.session_state.hist = train(
            st.session_state.data, hid, lay, epochs, lr, w_d, w_p, seed, prog, ph, param_psi, mono_w, reweight_h3, fix_E=fix_E)
        st.success("Training complete — check the **Results** tab.")
    if st.session_state.hist:
        h = st.session_state.hist
        f, a = plt.subplots(figsize=(8, 3.6), facecolor="none")
        a.plot(h["d"], color=CY, lw=2, label="L_data"); a.plot(h["p"], color=AM, lw=2, label="L_physics")
        a.set_yscale("log"); a.set_xlabel("epoch"); a.set_ylabel("loss (log)")
        a.legend(frameon=False); ax0(a); st.pyplot(f)
    elif not train_btn:
        st.info("Hit **Train PINN** in the sidebar.")

# ---------- 3 · RESULTS (base; truth-guarded, variable run count) ----------
with tb[3]:
    st.markdown("#### Inverse recovery")
    if st.session_state.nets and st.session_state.data:
        d, r = st.session_state.data, evaluate(st.session_state.nets, st.session_state.data)
        rel = lambda p, t: float(np.mean(np.abs(p - t) / (np.abs(t) + 1e-8)) * 100)
        comb = lambda K, h, e: K*h**3 + e
        truth = bool(d.get("has_truth", False)) and d.get("psi") is not None
        n = len(d["runs"])

        def h_err_for(i):
            run = d["runs"][i]
            if truth and run.get("h") is not None and len(run["h"]) == len(r["hs"][i]):
                return rel(r["hs"][i], run["h"]), "rec"          # synthetic: dense recovery
            with torch.no_grad():                                  # manual: fit / recovery @ data τ
                hd = st.session_state.nets["hn"][i](
                    torch.tensor(run["tau_s"], dtype=torch.float32).reshape(-1, 1), 1.0).numpy().flatten()
            if run.get("h") is not None:
                return rel(hd, run["h"]), "rec@data"
            return rel(hd, run["h_meas"]), "fit"

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Ψ(τ) error", f"{rel(r['psi'], d['psi']):.1f}%" if truth else "—")
        m2.metric("E(τ) error", f"{rel(r['e'], d['e']):.1f}%" if truth else "—")
        he0, lab0 = h_err_for(0)
        m3.metric(f"h run A ({lab0})", f"{he0:.2f}%")
        if n >= 2:
            he1, lab1 = h_err_for(1); m4.metric(f"h run B ({lab1})", f"{he1:.2f}%")
        else:
            m4.metric("h run B", "—")

        c1, c2 = st.columns(2)
        with c1:
            f, a = plt.subplots(figsize=(6, 3.6), facecolor="none")
            if truth:
                a.plot(d["tau"], d["psi"], color=CY, lw=2.4, label="true Ψ")
                a.plot(d["tau"], d["e"], color=AM, lw=2.4, label="true E")
            a.plot(d["tau"], r["psi"], color=CY, lw=2, ls="--", label="pred Ψ")
            a.plot(d["tau"], r["e"], color=AM, lw=2, ls="--", label="pred E")
            a.set_xlabel("τ"); a.legend(frameon=False, fontsize=8); ax0(a)
            a.set_title("Shared Ψ & E " + ("recovery" if truth else "(no ground truth)")); st.pyplot(f)
        with c2:
            f, a = plt.subplots(figsize=(6, 3.6), facecolor="none")
            for i in range(n):
                cp = comb(r["Ks"][i], r["hs"][i], r["e"])
                if truth and d.get("K_true") is not None:
                    ct = comb(d["K_true"][i], d["runs"][i]["h"], d["e"])
                    a.plot(d["tau"], ct, color=[CY, AM][i % 2], lw=2.4, label=f"true run{i}")
                a.plot(d["tau"], cp, color=[CY, AM][i % 2], lw=2, ls="--", label=f"pred run{i}")
            a.set_xlabel("τ"); a.set_ylabel("Kh³+E"); a.legend(frameon=False, fontsize=8); ax0(a)
            a.set_title("Combined ODE term (the identifiable part)"); st.pyplot(f)

        # ---- algebraic hybrid recovery + information-horizon leverage (ideas 2 & 1) ----
        tau_d = np.asarray(d["tau"], dtype=float)
        st.markdown("**Algebraic (hybrid) recovery** — Ψ, Ẽ solved pointwise from the two "
                    "smoothed thickness curves (the PINN denoises; the split is a direct solve, "
                    "so no mirror image). The curve ends where the two-run leverage vanishes.")
        w0 = d["runs"][0]["w"]; w1 = d["runs"][1]["w"] if n >= 2 else w0
        alg = None
        if n >= 2:
            try:
                alg = algebraic_split(st.session_state.nets["hn"], w0, w1, tau_d)
            except Exception as ex:
                st.warning("algebraic split failed: %r" % ex)
        if alg is not None:
            ca1, ca2 = st.columns(2)
            with ca1:
                f, a = plt.subplots(figsize=(6, 3.4), facecolor="none")
                if truth:
                    a.plot(tau_d, d["psi"], color=CY, lw=2.4, label="true Ψ")
                a.plot(alg["tau"], alg["Psi"], "--", color=RD if not truth else AM, lw=2,
                       label="algebraic Ψ")
                a.set_ylim(-0.5, max(2.0, np.nanmax(np.abs(alg["Psi"])) * 1.1))
                a.set_title("Ψ from direct 2x2 solve"); a.legend(frameon=False, fontsize=8)
                a.set_xlabel("τ"); ax0(a); st.pyplot(f)
            with ca2:
                f, a = plt.subplots(figsize=(6, 3.4), facecolor="none")
                if truth:
                    a.plot(tau_d, d["e"], color=CY, lw=2.4, label="true Ẽ")
                a.plot(alg["tau"], alg["E"], "--", color=RD if not truth else AM, lw=2,
                       label="algebraic Ẽ")
                a.set_title("Ẽ from direct 2x2 solve"); a.legend(frameon=False, fontsize=8)
                a.set_xlabel("τ"); ax0(a); st.pyplot(f)
            if truth:
                early = tau_d < 0.2
                m_e = early & ~np.isnan(alg["Psi"])
                m_f = ~np.isnan(alg["Psi"])
                e_err = rel(alg["Psi"][m_e], d["psi"][m_e]) if m_e.any() else float("nan")
                f_err = rel(alg["Psi"][m_f], d["psi"][m_f]) if m_f.any() else float("nan")
                st.caption("Algebraic Ψ — early window (τ<0.2) err **%.1f%%** vs full-domain "
                           "**%.1f%%**. The gap is the information horizon made visible: the direct "
                           "solve is accurate where leverage exists and *undefined* (masked) where it "
                           "doesn't. Notice there is no mirror image — this is a solve, not gradient "
                           "descent, so the Layer-1 amplitude/shape pathology is gone; only the "
                           "Layer-2 decay (late window) stays out of reach." % (e_err, f_err))
            else:
                st.caption("Direct solve of Ψ, Ẽ implied by your two curves. Where |D|≈0 (late τ) "
                           "the split is masked. If the consistency FLAG is up, this curve need not be "
                           "physical (negative / non-decaying) — that *is* the inconsistency drawn as a "
                           "line, which is the point.")
            # leverage / horizon plot (works in both modes; from the recovered h)
            h0r, h1r = r["hs"][0], r["hs"][1]
            Dlev = np.abs((w0 ** 2) * h0r ** 3 - (w1 ** 2) * h1r ** 3)
            tot = float(np.trapezoid(Dlev, tau_d)); e2 = tau_d < 0.2
            frac = float(np.trapezoid(Dlev[e2], tau_d[e2]) / tot) if tot > 0 else float("nan")
            f, a = plt.subplots(figsize=(7, 3.2), facecolor="none")
            a.plot(tau_d, Dlev, color="#cbd5e1", lw=2)
            a.axvline(0.2, color=RD, ls=":")
            a.set_title("Two-run leverage c(τ)=|w0²h0³−w1²h1³| · %.0f%% lives in τ<0.2"
                        % (100 * frac)); a.set_xlabel("τ"); ax0(a); st.pyplot(f)
            st.caption("This is the experimental-design map (idea 1): the red line at τ=0.2 bounds "
                       "the region that carries almost all the viscosity information. Sampling densely "
                       "left of that line — not evenly across [0,1] — is the only thing that can pull "
                       "the decay rate back into reach.")

        if not truth:
            # real per-run h-fit errors (what the verdict should read, not "by construction")
            param = st.session_state.nets
            runs = d["runs"]
            fit_errs = []
            for i, r in enumerate(runs):
                with torch.no_grad():
                    hp = param['hn'][i](r['td'], 1.0).numpy().ravel()
                fit_errs.append(float(np.mean(np.abs(hp - r['h_meas']) / (np.abs(r['h_meas']) + 1e-8)) * 100))

            # ms and hor for multi-start analysis (single-run fallback)
            ms = None  # multi-start spread not computed in single-run mode
            hor = {'frac_early': frac} if 'frac' in dir() else None

            v, arel, drel, note, level = verdict(ms, hor, fit_errs=fit_errs,
                                                 Ld_A=param.get('Ld_A'), Ld_C=param.get('Ld_C'))
            flag = {'ok': '✅', 'warn': '⚠️', 'bad': '🚨'}[level]
            status_card(level, "%s Trust verdict" % flag,
                        ["%s: %s" % (k, val) for k, val in v.items()] + [note])
    else:
        st.info("Generate or load data, then train.")

# ---------- 4 · MANUAL / CSV (ADDED TAB) ----------
with tb[4]:
    st.markdown("#### Manual / CSV input")
    st.caption("Give thickness-vs-time for one or more spin runs. Required columns: **t, h**. "
               "Optional: **run_id** (defaults to 0), **rpm** (needed for the multi-run ω² lever), "
               "**h_true** (to validate the thickness fit). If your values are dimensional, set "
               "**h_wet** and **t_ref** below so that h = h/h_wet (⇒ h(0)=1) and τ = t/t_ref (⇒ τ∈[0,1]).")

    with st.expander("normalisation (h_wet, t_ref, default RPM)", expanded=True):
        nc1, nc2 = st.columns(2)
        with nc1:
            h_wet = st.number_input("h_wet (same units as h)", 1e-9, 1e9, 1.0, format="%.4f",
                                    help="thickness at t=0; if your data is already dimensionless (h starts at 1) leave 1.0")
        with nc2:
            t_ref = st.number_input("t_ref (same units as t)", 1e-9, 1e9, 1.0, format="%.4f",
                                    help="pick so τ = t/t_ref spans ~[0,1]; if τ already in [0,1] leave 1.0")
        default_rpm = st.number_input("default RPM (used when a run has no rpm column)", 100, 10000, 3000, 100)

    st.markdown("**Option 1 — upload a CSV file**")
    upl = st.file_uploader("CSV file", type=["csv"], key="csvfile")

    st.markdown("**Option 2 — paste CSV text**")
    txt = st.text_area(" ", value="", height=170, key="csvtxt", placeholder=_MANUAL_EXAMPLE)
    if st.button("load example into the paste box", key="ldex"):
        st.session_state["csvtxt"] = _MANUAL_EXAMPLE; st.rerun()

    st.markdown("**Option 3 — type / edit a table**")
    df0 = st.session_state.get("df0")
    if df0 is None:
        df0 = pd.DataFrame(columns=["run_id", "t", "h", "rpm"], index=range(4)).astype(
            {"run_id": "float", "t": "float", "h": "float", "rpm": "float"})
    df = st.data_editor(df0, num_rows="dynamic", use_container_width=True, key="dfedit", hide_index=True)
    st.session_state["df0"] = df

    if st.button("Parse & load this data", use_container_width=True, key="parsebtn"):
        text = None
        if upl is not None:
            text = upl.getvalue().decode("utf-8-sig")
        elif st.session_state.get("csvtxt", "").strip():
            text = st.session_state["csvtxt"]
        elif df is not None and len(df):
            text = df.to_csv(index=False)
        if not text:
            st.error("Provide a CSV file, pasted text, or table rows first.")
        else:
            rows = _parse_rows(text)
            if not rows:
                st.error("No readable rows found — the header must contain at least **t** and **h**.")
            else:
                mdata = _build_manual(rows, h_wet, t_ref, default_rpm)
                if mdata is None:
                    st.error("Could not build runs (each run needs ≥ 2 points and h_wet, t_ref > 0).")
                else:
                    st.session_state.data = mdata
                    st.session_state.nets = st.session_state.hist = None
                    rps = sorted({r["rpm"] for r in mdata["runs"]})
                    if len({r["w"] for r in mdata["runs"]}) < 2:
                        st.warning("All runs share the same ω² scaling → no multi-run lever on Ψ. "
                                   "Add differing **rpm** values per run to break the Ψ/E degeneracy.")
                    st.success(f"Loaded {len(mdata['runs'])} run(s) · rpms={rps} · "
                               f"rpm_ref={mdata['manual_meta']['rpm_ref']}. Now hit ** Train PINN**.")
                    prev = []
                    for i, r in enumerate(mdata["runs"]):
                        for ts, hs in zip(r["tau_s"], r["h_meas"]):
                            prev.append({"run": i, "τ": round(float(ts), 3),
                                         "h": round(float(hs), 3), "w": round(float(r["w"]), 3)})
                    st.dataframe(prev, use_container_width=True, hide_index=True)

    if isinstance(st.session_state.get("data"), dict) and st.session_state["data"].get("manual_meta"):
        st.info("Manual data is currently loaded. (Switching to **Synthetic** and pressing "
                "** Generate data** will replace it.)")

# ---------- 5 · MODEL (base) ----------
with tb[5]:
    st.markdown("#### The model")
    st.latex(r"\frac{d\tilde h}{d\tau} \;=\; -\,\tilde K(\tau)\,\tilde h^{3} \;-\; \tilde E(\tau),"
             r"\qquad \tilde K(\tau)=\Big(\tfrac{\omega}{\omega_{\mathrm{ref}}}\Big)^{2}\Psi(\tau)")
    st.markdown("""
    - **h(τ)** — dimensionless film thickness, **Ψ(τ)** shared latent convective term, **E(τ)** evaporation.
    - Two runs at different RPM share **Ψ** and **E**; only the ω² scaling of K differs → this is what makes the
      inverse problem *identifiable in principle*.
    - The PINN enforces the ODE at unlabeled collocation points (physics loss) while fitting sparse thickness
      data (data loss). The thickness net uses a hard ansatz h = 1 − τ·softplus(·) so h(0)=1 exactly.
    - **Known failure mode:** Ψ and E are individually hard to disentangle — the optimizer can trade one against
      the other while still matching h(τ) almost perfectly. Watch the **Results** tab: a small h error alongside a
      large Ψ error is the *expected* non-identifiability signature, not a training bug.
    - **Manual data:** with no hidden truth, the app reports data-fit quality and the combined ODE term only; the
      Ψ/E *split* remains the structural limit. To recover Ψ decay you need dense *early*-time samples (τ≲0.2) or a
      viscosity-sensitive measurement.
    """)

# ═══════════════════ FIXES LAB (paste after Model tab) ═══════════════════
def train_cfg(data, cfg, hid, lay, epochs, lr, w_d, w_p, w_m, seed, prog=None, ph=None, tag=""):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    runs = data["runs"]; h_nets = [ThicknessNet(hid, lay) for _ in runs]
    e_init = estimate_E_late_sweep(data) if cfg.get("autofill") else None
    if e_init is None: e_init = (1.0, 1.0)
    psi = PsiParNet() if cfg.get("psipar") else PsiNet(hid, lay)
    em = cfg.get("emode", "free")
    e_net = EConstNet(e_init[0]) if em=="const" else (EExpNet(*e_init) if em=="exp" else ETildeNet(hid, lay))
    TD, HD, TC = [], [], []
    for r in runs:
        ts, hs = r["tau_s"], r["h_meas"]
        if cfg.get("early") and r.get("h") is not None:      # dense early measurements
            te = np.sort(rng.uniform(0.0, 0.2, 6)); he = np.interp(te, data["tau"], r["h"])
            he = np.clip(he + rng.normal(0, float(noise), len(te))*he, 1e-4, None)
            ts = np.concatenate([ts, te]); hs = np.concatenate([hs, he])
        TD.append(torch.tensor(ts, dtype=torch.float32).reshape(-1,1))
        HD.append(torch.tensor(hs, dtype=torch.float32).reshape(-1,1))
        n = int(n_colloc)
        tc = (np.sort(np.concatenate([rng.uniform(0,1,n//2), rng.beta(2,5,n-n//2)]))
              if cfg.get("early") else np.sort(rng.uniform(0,1,n)))
        TC.append(torch.tensor(tc, dtype=torch.float32).reshape(-1,1).requires_grad_(True))
    params = [p for net in h_nets for p in net.parameters()] + list(psi.parameters()) + list(e_net.parameters())
    opt = optim.Adam(params, lr=lr); t0 = time.time()
    for ep in range(epochs):
        opt.zero_grad(); Ld = Lp = Lm = 0.0
        for i, r in enumerate(runs):
            Ld = Ld + torch.mean((h_nets[i](TD[i], 1.0) - HD[i])**2)
            res, hc, _ = residual(h_nets[i], psi, e_net, TC[i], r["w"])
            if cfg.get("rw"):                                 # 1/h³ reweight (capped)
                wgt = torch.clamp(1.0/(hc.detach()**3 + 1e-4), max=100.0)
                res = res * (wgt / wgt.mean())
            Lp = Lp + torch.mean(res**2)
            if cfg.get("mono") and not cfg.get("psipar"):     # Ψ monotone-decay penalty
                dp = torch.autograd.grad(psi(TC[i]), TC[i], torch.ones_like(TC[i]), create_graph=True)[0]
                Lm = Lm + torch.mean(torch.relu(dp)**2)
        loss = w_d*Ld + w_p*Lp + w_m*Lm
        loss.backward(); opt.step()
        if prog is not None and ep % 50 == 0: prog.progress((ep+1)/epochs)
        if ph is not None and ep % 50 == 0:
            lmv = Lm.item() if isinstance(Lm, torch.Tensor) else Lm
            ph.caption(f"{tag} · epoch {ep+1}/{epochs} · Ld {Ld.item():.4f} · Lp {Lp.item():.4f} · Lm {lmv:.4f}")
    return dict(h_nets=h_nets, psi=psi, e=e_net), time.time()-t0

def sweep_metrics(nets, data):
    r = evaluate(nets, data); m = {}
    rel = lambda p, t: float(np.mean(np.abs(p-t)/(np.abs(t)+1e-8))*100)
    if data.get("psi") is not None:
        m["psi"] = rel(r["psi"], data["psi"]); m["E"] = rel(r["e"], data["e"])
        m["h"] = float(np.mean([rel(r["hs"][i], data["runs"][i]["h"]) for i in range(len(data["runs"]))]))
        m["comb"] = float(np.mean([rel(r["Ks"][i]*r["hs"][i]**3 + r["e"],
                                       data["K_true"][i]*data["runs"][i]["h"]**3 + data["e"])
                                   for i in range(len(data["runs"]))]))
    return m

def individual_configs():
    yield ("baseline",            {})
    yield ("dense-early",         dict(early=True))
    yield ("PsiPar",              dict(psipar=True))
    yield ("mono",                dict(mono=True))
    yield ("1/h³ reweight",       dict(rw=True))
    yield ("E=const",             dict(emode="const"))
    yield ("E=exp",               dict(emode="exp"))
    yield ("E=exp+autofill",      dict(emode="exp", autofill=True))

def exhaustive_configs():   # 60 pruned combos
    for early, psipar, rw in itertools.product([False, True], repeat=3):
        for mono in ([False] if psipar else [False, True]):
            for emode in ("free", "const", "exp"):
                for autofill in ([False] if emode == "free" else [False, True]):
                    cfg = dict(early=early, psipar=psipar, mono=mono, rw=rw, emode=emode, autofill=autofill)
                    yield (cfg_name(cfg), cfg)

def curated_configs():      # sensible stacks, incl. two "full" stacks
    for c in individual_configs(): yield c
    yield ("early+rw",                dict(early=True, rw=True))
    yield ("early+psipar",            dict(early=True, psipar=True))
    yield ("psipar+exp+autofill",     dict(psipar=True, emode="exp", autofill=True))
    yield ("early+mono+rw",           dict(early=True, mono=True, rw=True))
    yield ("early+rw+exp+autofill",   dict(early=True, rw=True, emode="exp", autofill=True))
    yield ("full(mono)",              dict(early=True, mono=True, rw=True, emode="exp", autofill=True))
    yield ("full(psipar)",            dict(early=True, psipar=True, rw=True, emode="exp", autofill=True))

def cfg_name(cfg):
    p = [k for k in ("early","psipar","mono","rw","autofill") if cfg.get(k)]
    if cfg.get("emode","free") != "free": p.append("E="+cfg["emode"])
    return "+".join(p) if p else "baseline"

def run_sweep(configs, epochs, w_m):
    configs = list(configs)
    st.session_state.setdefault("sweep_rows", [])
    prg = st.progress(0.0); ph = st.empty(); slot = st.empty()
    t_first = None
    for k, (name, cfg) in enumerate(configs):
        prg.progress(k/len(configs))
        nets, dt = train_cfg(st.session_state.data, cfg, hid, lay, epochs, lr, w_d, w_p, w_m,
                             seed, prog=None, ph=ph, tag=name)
        m = sweep_metrics(nets, st.session_state.data)
        st.session_state.sweep_rows.append(
            {"config": name, "Ψ%": round(m["psi"],1) if "psi" in m else None,
             "E%": round(m["E"],1) if "E" in m else None,
             "comb%": round(m["comb"],1) if "comb" in m else None,
             "h%": round(m["h"],1) if "h" in m else None, "sec": round(dt)})
        key = m.get("psi", m.get("comb", 1e9))
        best = st.session_state.get("sweep_best")
        if best is None or key < best[0]: st.session_state.sweep_best = (key, name, nets)
        if t_first is None: t_first = dt
        ph.caption(f"[{k+1}/{len(configs)}] {name} · {dt:.0f}s · ≈{t_first*(len(configs)-k-1)/60:.0f} min left")
        slot.dataframe(pd.DataFrame(st.session_state.sweep_rows), use_container_width=True)
    prg.progress(1.0); ph.caption("sweep complete")

with tb[6]:
    st.markdown("#### Fixes Lab — individual & combined mitigation sweeps")
    st.caption("Needs loaded data (synthetic recommended; manual data has no Ψ/E truth to score against). "
               "Exhaustive = all 60 pruned combinations — heavy on CPU; keep combo epochs low.")
    c1, c2, c3, c4 = st.columns(4)
    with c1: ep_ind = st.number_input("epochs · individual", 200, 4000, 2000, 100)
    with c2: ep_swp = st.number_input("epochs · combo", 200, 4000, 800, 100)
    with c3: w_mono = st.slider("mono weight", 0.0, 5.0, 1.0, 0.1)
    with c4: exhaustive = st.checkbox("exhaustive combos (60)", value=False)
    if exhaustive:
        st.warning(f"60 configs × {ep_swp} epochs. ETA is shown live after the first config; "
                   "on CPU this can take 10–30+ min and the UI will look frozen. Consider ≤800 epochs.")
    b_ind = st.button("Run individual fixes", use_container_width=True, key="sw_ind")
    b_cmb = st.button("Run combination sweep (separate)", use_container_width=True, key="sw_combo")
    if b_ind or b_cmb:
        if st.session_state.data is None:
            st.warning("Load/generate data first."); st.stop()
        if b_ind: run_sweep(individual_configs(), int(ep_ind), w_mono)
        else:     run_sweep(exhaustive_configs() if exhaustive else curated_configs(), int(ep_swp), w_mono)
        st.rerun()
    if st.session_state.get("sweep_rows"):
        df = pd.DataFrame(st.session_state.sweep_rows)
        if "Ψ%" in df.columns: df = df.sort_values("Ψ%")
        st.dataframe(df, use_container_width=True)
        bb = st.columns(3)
        if bb[0].button("Load best sweep into Results tab", key="ldbest") and st.session_state.get("sweep_best"):
            st.session_state.nets = st.session_state.sweep_best[2]
            st.success(f"Loaded best: {st.session_state.sweep_best[1]}")
        if bb[1].button("Clear sweep results", key="clrsw"):
            st.session_state.pop("sweep_rows", None); st.session_state.pop("sweep_best", None); st.rerun()
        st.caption(f"Best so far: {st.session_state.get('sweep_best', (None,'—'))[1]}")
