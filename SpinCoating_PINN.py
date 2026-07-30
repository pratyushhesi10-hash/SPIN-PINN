# app.py — SpinCoat PINN Lab  (+ Manual / CSV data tab)
# Run:  streamlit run app.py
import io, csv
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

# ─────────────────────────── Page & theme (unchanged base) ───────────────────────────
st.set_page_config(page_title="SpinCoat PINN Lab", page_icon="🌀", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
html,body,[class*="css"],.stMarkdown{font-family:'IBM Plex Sans',sans-serif;}
[data-testid="stAppViewContainer"]{
  background:
   radial-gradient(1100px 700px at 88% -8%, rgba(34,211,238,.08), transparent 60%),
   radial-gradient(900px 650px at -8% 108%, rgba(251,191,36,.06), transparent 60%),
   radial-gradient(800px 800px at 50% 130%, rgba(56,189,248,.05), transparent 60%),
   #0b1220;}
[data-testid="stAppViewContainer"]::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:0;
  background-image:radial-gradient(rgba(148,163,184,.08) 1px, transparent 1px);background-size:26px 26px;}
[data-testid="stMain"],[data-testid="stMainBlockContainer"]{background:transparent;}
[data-testid="stSidebar"]{background:rgba(13,20,32,.78);border-right:1px solid rgba(148,163,184,.12);}
.hero{padding:6px 0 2px;}
.hero-top{display:flex;align-items:center;gap:16px;}
.spin{font-size:44px;display:inline-block;animation:spin 7s linear infinite;}
@keyframes spin{from{transform:rotate(0)}to{transform:rotate(360deg)}}
.kicker{font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:.22em;color:#22d3ee;text-transform:uppercase;}
.title{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:44px;line-height:1.05;margin:2px 0 0;color:#eef3fb;}
.title .accent{color:#fbbf24;}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px;}
.chip{font-family:'IBM Plex Mono',monospace;font-size:11.5px;padding:5px 11px;border-radius:999px;
  border:1px solid rgba(148,163,184,.28);color:#cbd5e1;background:rgba(148,163,184,.08);transition:.2s;}
.chip:hover{border-color:#22d3ee;color:#67e8f9;transform:translateY(-1px);}
.chip-cyan{border-color:rgba(34,211,238,.45);color:#67e8f9;background:rgba(34,211,238,.10);}
.chip-amber{border-color:rgba(251,191,36,.45);color:#fcd34d;background:rgba(251,191,36,.10);}
.stTabs [data-baseweb="tab-list"]{gap:6px;border-bottom:1px solid rgba(148,163,184,.18);}
.stTabs [data-baseweb="tab"]{font-family:'Space Grotesk',sans-serif;font-weight:600;}
[data-testid="stMetric"]{background:rgba(148,163,184,.07);border:1px solid rgba(148,163,184,.16);
  border-radius:14px;padding:12px 16px;transition:.25s;}
[data-testid="stMetric"]:hover{border-color:rgba(34,211,238,.5);transform:translateY(-2px);}
[data-testid="stMetricLabel"]{color:#94a3b8;}
[data-testid="stMetricValue"]{font-family:'Space Grotesk',sans-serif;color:#eef3fb;}
.stButton>button{border-radius:12px;font-family:'Space Grotesk',sans-serif;font-weight:600;transition:.2s;}
.stButton>button:hover{transform:translateY(-2px);}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="hero-top"><span class="spin">🌀</span>
    <div><div class="kicker">Physics-Informed Neural Network · Inverse Discovery</div>
    <h1 class="title">SpinCoat <span class="accent">PINN</span> Lab</h1></div></div>
  <div class="chips">
    <span class="chip">dĥ/dτ = −K̃(τ)·ĥ³ − Ẽ(τ)</span>
    <span class="chip chip-cyan">2 spin runs · shared Ψ &amp; Ẽ</span>
    <span class="chip chip-amber">synthetic or YOUR data</span>
  </div>
</div>
""", unsafe_allow_html=True)

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

class ETildeNet(nn.Module):
    def __init__(self, h=32, L=3):
        super().__init__()
        L_ = [nn.Linear(1, h), nn.Tanh()]
        for _ in range(L - 1): L_ += [nn.Linear(h, h), nn.Tanh()]
        L_ += [nn.Linear(h, 1)]
        self.net = nn.Sequential(*L_); self.sp = nn.Softplus()
    def forward(self, tau): return self.sp(self.net(tau))

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
def generate_data(psi_A, psi_d, E_B, E_d, rpm_a, rpm_b, n_meas, noise, n_colloc, seed):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    tau = np.linspace(0, 1, 500); w_ref = rpm_a
    runs, K_true = [], []
    for rpm in (rpm_a, rpm_b):
        w = rpm / w_ref
        h = simulate(psi_A, psi_d, E_B, E_d, w, tau)
        K_true.append((w**2) * psi_A*np.exp(-psi_d*tau))
        edges = np.linspace(0, 1, n_meas + 1)
        tgt = np.array([rng.uniform(edges[i], edges[i+1]) for i in range(n_meas)])
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
def train(data, h, L, epochs, lr, w_d, w_p, seed, prog, ph):
    torch.manual_seed(seed)
    h_nets = [ThicknessNet(h, L) for _ in data["runs"]]
    psi, e = PsiNet(h, L), ETildeNet(h, L)
    p = [p for n in h_nets for p in n.parameters()] + list(psi.parameters()) + list(e.parameters())
    opt = optim.Adam(p, lr=lr); hist = dict(d=[], p=[], t=[])
    for ep in range(epochs):
        opt.zero_grad(); Ld = Lp = 0.0
        for i, r in enumerate(data["runs"]):                       # enumerate (was .index(r))
            td = torch.tensor(r["tau_s"], dtype=torch.float32).reshape(-1, 1)
            hd = torch.tensor(r["h_meas"], dtype=torch.float32).reshape(-1, 1)
            Ld = Ld + torch.mean((h_nets[i](td, 1.0) - hd)**2)
            tc = torch.tensor(r["tau_c"], dtype=torch.float32).reshape(-1, 1).requires_grad_(True)
            res, _, _ = residual(h_nets[i], psi, e, tc, r["w"])
            Lp = Lp + torch.mean(res**2)
        loss = w_d*Ld + w_p*Lp; loss.backward(); opt.step()
        hist["d"].append(Ld.item()); hist["p"].append(Lp.item()); hist["t"].append(loss.item())
        if ep % 20 == 0 or ep == epochs - 1:
            prog.progress((ep+1)/epochs)
            ph.caption(f"epoch {ep+1}/{epochs} · L_data {Ld.item():.5f} · L_phys {Lp.item():.5f}")
    return dict(h_nets=h_nets, psi=psi, e=e), hist

def evaluate(nets, data):
    with torch.no_grad():
        t = torch.tensor(data["tau"], dtype=torch.float32).reshape(-1, 1)
        psi, e = nets["psi"](t).numpy().flatten(), nets["e"](t).numpy().flatten()
        hs = [n(t, 1.0).numpy().flatten() for n in nets["h_nets"]]
    return dict(psi=psi, e=e, hs=hs, Ks=[(r["w"]**2)*psi for r in data["runs"]])

# ─────────────────────────── Plot helper (base) ───────────────────────────
plt.rcParams.update({"figure.facecolor": "none", "axes.facecolor": "none",
                     "axes.edgecolor": "#334155", "axes.labelcolor": "#cbd5e1",
                     "text.color": "#cbd5e1", "axes.grid": True, "grid.color": "#1e293b",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "IBM Plex Sans"})
CY, AM = "#22d3ee", "#fbbf24"
def ax0(a): a.tick_params(colors="#94a3b8"); a.grid(alpha=.25); return a

# ─────────────────────────── Sidebar (base + data-source toggle) ───────────────────────────
st.sidebar.markdown("### ⚙️ Controls")
source = st.sidebar.radio("Data source", ["Synthetic", "Manual / CSV"], index=0,
                          help="Synthetic = the app generates the hidden truth. "
                               "Manual / CSV = your own thickness-vs-time data.")
SRC_SYN = (source == "Synthetic")

with st.sidebar.expander("🧪 Physics", expanded=True):
    psi_A = st.slider("Ψ_A · convective strength", 0.1, 3.0, 1.2, 0.05)
    psi_d = st.slider("Ψ decay", 0.5, 6.0, 3.0, 0.1)
    E_B   = st.slider("E_B · evaporation strength", 0.5, 6.0, 3.0, 0.1)
    E_d   = st.slider("E decay", 0.5, 6.0, 3.5, 0.1)
    rpm_a = st.slider("Run A · RPM", 1000, 6000, 3000, 100)
    rpm_b = st.slider("Run B · RPM", 1000, 6000, 4500, 100)
with st.sidebar.expander("🧫 Synthetic data", expanded=SRC_SYN):
    n_meas  = st.slider("Measurements / run", 4, 24, 8)
    noise   = st.slider("Noise σ", 0.0, 0.10, 0.02, 0.005)
    n_colloc = st.slider("Collocation points", 50, 400, 200, 10)
    seed = st.number_input("Seed", 0, 999, 42)
with st.sidebar.expander("🧠 Training", expanded=True):
    epochs = st.slider("Epochs", 200, 4000, 1500, 100)
    lr = st.select_slider("Learning rate", [5e-4, 1e-3, 2e-3, 5e-3], value=1e-3)
    hid = st.slider("Hidden width", 16, 64, 32, 8)
    lay = st.slider("Hidden layers", 2, 5, 3)
    w_d = st.slider("W_data", 0.1, 5.0, 1.0, 0.1)
    w_p = st.slider("W_physics", 0.1, 5.0, 1.0, 0.1)

st.sidebar.markdown("---")
if SRC_SYN:
    gen_btn = st.sidebar.button("📡 Generate data", use_container_width=True)
else:
    st.sidebar.caption("Load your thickness data in the **✎ Manual / CSV** tab, then Train.")
    gen_btn = False
train_btn = st.sidebar.button("🧠 Train PINN", use_container_width=True, type="primary")

for k in ("data", "nets", "hist"):
    st.session_state.setdefault(k, None)
if gen_btn:
    st.session_state.data = generate_data(psi_A, psi_d, E_B, E_d, rpm_a, rpm_b, n_meas, noise, n_colloc, seed)
    st.session_state.nets = st.session_state.hist = None

# ─────────────────────────── Tabs (base 5 + added Manual) ───────────────────────────
tb = st.tabs(["🧪 Physics", "📡 Data", "🧠 Train", "📊 Results", "✎ Manual / CSV", "ℹ️ Model"])

# ---------- 0 · PHYSICS (base; note when manual) ----------
with tb[0]:
    st.markdown("#### Live thinning simulator")
    if not SRC_SYN:
        st.caption("ℹ️ This simulator draws the *synthetic* physics from the sliders, for intuition. "
                   "Your loaded data drives the Data / Train / Results tabs.")
    tau = np.linspace(0, 1, 500); w_ref = rpm_a
    c1, c2 = st.columns(2)
    with c1:
        f, a = plt.subplots(figsize=(6, 3.6), facecolor="none")
        for rpm, c in ((rpm_a, CY), (rpm_b, AM)):
            a.plot(tau, simulate(psi_A, psi_d, E_B, E_d, rpm/w_ref, tau), color=c, lw=2.4, label=f"{rpm} RPM")
        a.set_xlabel("τ"); a.set_ylabel("ĥ(τ)"); a.set_title("Film thinning ĥ(τ)")
        a.legend(frameon=False); ax0(a); st.pyplot(f)
    with c2:
        f, a = plt.subplots(figsize=(6, 3.6), facecolor="none")
        a.plot(tau, psi_A*np.exp(-psi_d*tau), color=CY, lw=2.4, label="Ψ(τ)")
        a.plot(tau, E_B*np.exp(-E_d*tau), color=AM, lw=2.4, label="Ẽ(τ)")
        a.set_xlabel("τ"); a.set_title("Latent Ψ(τ) & evaporation Ẽ(τ)")
        a.legend(frameon=False); ax0(a); st.pyplot(f)
    st.caption("K̃(τ) = (ω/ω_ref)²·Ψ(τ) — higher spin → stronger convective thinning.")

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
                    a.plot(d["tau"], run["h"], color=[CY, AM][i % 2], lw=2.2, alpha=.5, label="true ĥ(τ)")
                elif run.get("h") is not None:
                    a.plot(run["tau_s"], run["h"], color=[CY, AM][i % 2], lw=2.2, alpha=.5, label="true ĥ @data")
                a.scatter(run["tau_s"], run["h_meas"], color=[CY, AM][i % 2], s=46, zorder=5, label="data")
                a.scatter(run["tau_c"], np.zeros_like(run["tau_c"]),
                          marker="|", color="#64748b", s=60, label="collocation")
                a.set_xlabel("τ"); a.set_ylabel("ĥ"); a.set_title(f"Run {i} · {run.get('rpm', '?')} RPM")
                a.legend(frameon=False, fontsize=8); ax0(a); st.pyplot(f)
        if n > 2:
            st.caption(f"(showing first 2 of {n} runs)")
        if SRC_SYN:
            st.caption(f"{n_meas} stratified samples/run · σ={noise} · {n_colloc} collocation points · seed {seed}")
        else:
            m = d.get("manual_meta", {})
            st.caption(f"manual · {n} run(s) · h_wet={m.get('h_wet')} · t_ref={m.get('t_ref')} · rpm_ref={m.get('rpm_ref')}")
    else:
        st.info("Hit **📡 Generate data** (synthetic) or load data in the **✎ Manual / CSV** tab.")

# ---------- 2 · TRAIN (base; safe guard instead of silent synthetic) ----------
with tb[2]:
    st.markdown("#### Training")
    if train_btn:
        if st.session_state.data is None:
            st.warning("No data loaded yet. Use **📡 Generate data** (synthetic) or load CSV / manual "
                       "data in the **✎ Manual / CSV** tab first.")
            st.stop()
        prog = st.progress(0); ph = st.empty()
        st.session_state.nets, st.session_state.hist = train(
            st.session_state.data, hid, lay, epochs, lr, w_d, w_p, seed, prog, ph)
        st.success("Training complete ✅ — check the **Results** tab.")
    if st.session_state.hist:
        h = st.session_state.hist
        f, a = plt.subplots(figsize=(8, 3.6), facecolor="none")
        a.plot(h["d"], color=CY, lw=2, label="L_data"); a.plot(h["p"], color=AM, lw=2, label="L_physics")
        a.set_yscale("log"); a.set_xlabel("epoch"); a.set_ylabel("loss (log)")
        a.legend(frameon=False); ax0(a); st.pyplot(f)
    elif not train_btn:
        st.info("Hit **🧠 Train PINN** in the sidebar.")

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
                hd = st.session_state.nets["h_nets"][i](
                    torch.tensor(run["tau_s"], dtype=torch.float32).reshape(-1, 1), 1.0).numpy().flatten()
            if run.get("h") is not None:
                return rel(hd, run["h"]), "rec@data"
            return rel(hd, run["h_meas"]), "fit"

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Ψ(τ) error", f"{rel(r['psi'], d['psi']):.1f}%" if truth else "—")
        m2.metric("Ẽ(τ) error", f"{rel(r['e'], d['e']):.1f}%" if truth else "—")
        he0, lab0 = h_err_for(0)
        m3.metric(f"ĥ run A ({lab0})", f"{he0:.2f}%")
        if n >= 2:
            he1, lab1 = h_err_for(1); m4.metric(f"ĥ run B ({lab1})", f"{he1:.2f}%")
        else:
            m4.metric("ĥ run B", "—")

        c1, c2 = st.columns(2)
        with c1:
            f, a = plt.subplots(figsize=(6, 3.6), facecolor="none")
            if truth:
                a.plot(d["tau"], d["psi"], color=CY, lw=2.4, label="true Ψ")
                a.plot(d["tau"], d["e"], color=AM, lw=2.4, label="true Ẽ")
            a.plot(d["tau"], r["psi"], color=CY, lw=2, ls="--", label="pred Ψ")
            a.plot(d["tau"], r["e"], color=AM, lw=2, ls="--", label="pred Ẽ")
            a.set_xlabel("τ"); a.legend(frameon=False, fontsize=8); ax0(a)
            a.set_title("Shared Ψ & Ẽ " + ("recovery" if truth else "(no ground truth)")); st.pyplot(f)
        with c2:
            f, a = plt.subplots(figsize=(6, 3.6), facecolor="none")
            for i in range(n):
                cp = comb(r["Ks"][i], r["hs"][i], r["e"])
                if truth and d.get("K_true") is not None:
                    ct = comb(d["K_true"][i], d["runs"][i]["h"], d["e"])
                    a.plot(d["tau"], ct, color=[CY, AM][i % 2], lw=2.4, label=f"true run{i}")
                a.plot(d["tau"], cp, color=[CY, AM][i % 2], lw=2, ls="--", label=f"pred run{i}")
            a.set_xlabel("τ"); a.set_ylabel("K̃ĥ³+Ẽ"); a.legend(frameon=False, fontsize=8); ax0(a)
            a.set_title("Combined ODE term (the identifiable part)"); st.pyplot(f)

        if not truth:
            st.caption("ℹ️ Manual data has no hidden Ψ/Ẽ to compare against, so Ψ/Ẽ tiles read “—” and "
                       "the ĥ tiles show the data-fit. The *combined* term’s shape is what the physics pins — "
                       "trust it (and Ẽ) more than Ψ alone; the Ψ/Ẽ split is the known identifiability limit.")
        st.caption("⚠️ Identifiability: K̃ and  trade off against each other — the *combined* term "
                   "K̃ĥ³+Ẽ is what the data actually pins down.")
    else:
        st.info("Generate or load data, then train.")

# ---------- 4 · MANUAL / CSV (ADDED TAB) ----------
with tb[4]:
    st.markdown("#### Manual / CSV input")
    st.caption("Give thickness-vs-time for one or more spin runs. Required columns: **t, h**. "
               "Optional: **run_id** (defaults to 0), **rpm** (needed for the multi-run ω² lever), "
               "**h_true** (to validate the thickness fit). If your values are dimensional, set "
               "**h_wet** and **t_ref** below so that ĥ = h/h_wet (⇒ ĥ(0)=1) and τ = t/t_ref (⇒ τ∈[0,1]).")

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

    if st.button("⬇  Parse & load this data", use_container_width=True, type="primary", key="parsebtn"):
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
                                   "Add differing **rpm** values per run to break the Ψ/Ẽ degeneracy.")
                    st.success(f"Loaded {len(mdata['runs'])} run(s) · rpms={rps} · "
                               f"rpm_ref={mdata['manual_meta']['rpm_ref']}. Now hit **🧠 Train PINN**.")
                    prev = []
                    for i, r in enumerate(mdata["runs"]):
                        for ts, hs in zip(r["tau_s"], r["h_meas"]):
                            prev.append({"run": i, "τ": round(float(ts), 3),
                                         "ĥ": round(float(hs), 3), "w": round(float(r["w"]), 3)})
                    st.dataframe(prev, use_container_width=True, hide_index=True)

    if isinstance(st.session_state.get("data"), dict) and st.session_state["data"].get("manual_meta"):
        st.info("Manual data is currently loaded. (Switching to **Synthetic** and pressing "
                "**📡 Generate data** will replace it.)")

# ---------- 5 · MODEL (base) ----------
with tb[5]:
    st.markdown("#### The model")
    st.latex(r"\frac{d\tilde h}{d\tau} \;=\; -\,\tilde K(\tau)\,\tilde h^{3} \;-\; \tilde E(\tau),"
             r"\qquad \tilde K(\tau)=\Big(\tfrac{\omega}{\omega_{\mathrm{ref}}}\Big)^{2}\Psi(\tau)")
    st.markdown("""
    - **ĥ(τ)** — dimensionless film thickness, **Ψ(τ)** shared latent convective term, **Ẽ(τ)** evaporation.
    - Two runs at different RPM share **Ψ** and **Ẽ**; only the ω² scaling of K̃ differs → this is what makes the
      inverse problem *identifiable in principle*.
    - The PINN enforces the ODE at unlabeled collocation points (physics loss) while fitting sparse thickness
      data (data loss). The thickness net uses a hard ansatz ĥ = 1 − τ·softplus(·) so ĥ(0)=1 exactly.
    - **Known failure mode:** Ψ and Ẽ are individually hard to disentangle — the optimizer can trade one against
      the other while still matching ĥ(τ) almost perfectly. Watch the **Results** tab: a small ĥ error alongside a
      large Ψ error is the *expected* non-identifiability signature, not a training bug.
    - **Manual data:** with no hidden truth, the app reports data-fit quality and the combined ODE term only; the
      Ψ/Ẽ *split* remains the structural limit. To recover Ψ decay you need dense *early*-time samples (τ≲0.2) or a
      viscosity-sensitive measurement.
    """)
