# app.py — SpinCoat PINN Lab
# Run:  streamlit run app.py
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────── Page & theme ───────────────────────────
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
  border:1px solid rgba(148,163,184,.28);color:#cbd5e1;background:rgba(148,163,184,.08);}
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
    <span class="chip chip-amber">sparse noisy ellipsometry</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────── Model (from your notebook) ───────────────────────────
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

# ─────────────────────────── Physics / data ───────────────────────────
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
                psi=psi_A*np.exp(-psi_d*tau), e=E_B*np.exp(-E_d*tau))

def train(data, h, L, epochs, lr, w_d, w_p, seed, prog, ph):
    torch.manual_seed(seed)
    h_nets = [ThicknessNet(h, L) for _ in data["runs"]]
    psi, e = PsiNet(h, L), ETildeNet(h, L)
    p = [p for n in h_nets for p in n.parameters()] + list(psi.parameters()) + list(e.parameters())
    opt = optim.Adam(p, lr=lr); hist = dict(d=[], p=[], t=[])
    for ep in range(epochs):
        opt.zero_grad(); Ld = Lp = 0.0
        for r in data["runs"]:
            td = torch.tensor(r["tau_s"], dtype=torch.float32).reshape(-1, 1)
            hd = torch.tensor(r["h_meas"], dtype=torch.float32).reshape(-1, 1)
            Ld = Ld + torch.mean((h_nets[data["runs"].index(r)](td, 1.0) - hd)**2)
            tc = torch.tensor(r["tau_c"], dtype=torch.float32).reshape(-1, 1).requires_grad_(True)
            res, _, _ = residual(h_nets[data["runs"].index(r)], psi, e, tc, r["w"])
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

# ─────────────────────────── Plot helpers ───────────────────────────
plt.rcParams.update({"figure.facecolor": "none", "axes.facecolor": "none",
                     "axes.edgecolor": "#334155", "axes.labelcolor": "#cbd5e1",
                     "text.color": "#cbd5e1", "axes.grid": True, "grid.color": "#1e293b",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "IBM Plex Sans"})
CY, AM = "#22d3ee", "#fbbf24"
def ax0(a): a.tick_params(colors="#94a3b8"); a.grid(alpha=.25); return a

# ─────────────────────────── Sidebar ───────────────────────────
st.sidebar.markdown("### ⚙️ Controls")
with st.sidebar.expander("🧪 Physics", expanded=True):
    psi_A = st.slider("Ψ_A · convective strength", 0.1, 3.0, 1.2, 0.05)
    psi_d = st.slider("Ψ decay", 0.5, 6.0, 3.0, 0.1)
    E_B   = st.slider("E_B · evaporation strength", 0.5, 6.0, 3.0, 0.1)
    E_d   = st.slider("E decay", 0.5, 6.0, 3.5, 0.1)
    rpm_a = st.slider("Run A · RPM", 1000, 6000, 3000, 100)
    rpm_b = st.slider("Run B · RPM", 1000, 6000, 4500, 100)
with st.sidebar.expander("📡 Synthetic data", expanded=True):
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

gen_btn  = st.sidebar.button("📡 Generate data", use_container_width=True)
train_btn = st.sidebar.button("🧠 Train PINN", use_container_width=True, type="primary")

for k in ("data", "nets", "hist"):
    st.session_state.setdefault(k, None)
if gen_btn:
    st.session_state.data = generate_data(psi_A, psi_d, E_B, E_d, rpm_a, rpm_b, n_meas, noise, n_colloc, seed)
    st.session_state.nets = st.session_state.hist = None

# ─────────────────────────── Tabs ───────────────────────────
tb = st.tabs(["🧪 Physics", "📡 Data", "🧠 Train", "📊 Results", "ℹ️ Model"])

with tb[0]:
    st.markdown("#### Live thinning simulator")
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

with tb[1]:
    st.markdown("#### Synthetic sparse measurements")
    if st.session_state.data:
        d = st.session_state.data
        c1, c2 = st.columns(2)
        for i, c in enumerate((CY, AM)):
            with c1 if i == 0 else c2:
                f, a = plt.subplots(figsize=(6, 3.6), facecolor="none")
                a.plot(d["tau"], d["runs"][i]["h"], color=c, lw=2.2, alpha=.5, label="true ĥ(τ)")
                a.scatter(d["runs"][i]["tau_s"], d["runs"][i]["h_meas"], color=c, s=46, zorder=5, label="sparse data")
                a.scatter(d["runs"][i]["tau_c"], np.zeros_like(d["runs"][i]["tau_c"]),
                          marker="|", color="#64748b", s=60, label="collocation")
                a.set_xlabel("τ"); a.set_ylabel("ĥ"); a.set_title(f"Run {i} · {d['runs'][i]['rpm']} RPM")
                a.legend(frameon=False, fontsize=8); ax0(a); st.pyplot(f)
        st.caption(f"{n_meas} stratified samples/run · σ={noise} · {n_colloc} collocation points · seed {seed}")
    else:
        st.info("Hit **📡 Generate data** in the sidebar.")

with tb[2]:
    st.markdown("#### Training")
    if train_btn:
        if st.session_state.data is None:
            st.session_state.data = generate_data(psi_A, psi_d, E_B, E_d, rpm_a, rpm_b, n_meas, noise, n_colloc, seed)
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
    else:
        st.info("Hit **🧠 Train PINN** in the sidebar.")

with tb[3]:
    st.markdown("#### Inverse recovery")
    if st.session_state.nets and st.session_state.data:
        d, r = st.session_state.data, evaluate(st.session_state.nets, st.session_state.data)
        rel = lambda p, t: float(np.mean(np.abs(p - t) / (np.abs(t) + 1e-8)) * 100)
        comb = lambda K, h, e: K*h**3 + e
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Ψ(τ) error", f"{rel(r['psi'], d['psi']):.1f}%")
        m2.metric("Ẽ(τ) error", f"{rel(r['e'], d['e']):.1f}%")
        m3.metric("ĥ run A err", f"{rel(r['hs'][0], d['runs'][0]['h']):.2f}%")
        m4.metric("ĥ run B err", f"{rel(r['hs'][1], d['runs'][1]['h']):.2f}%")
        c1, c2 = st.columns(2)
        with c1:
            f, a = plt.subplots(figsize=(6, 3.6), facecolor="none")
            a.plot(d["tau"], d["psi"], color=CY, lw=2.4, label="true Ψ")
            a.plot(d["tau"], r["psi"], color=CY, lw=2, ls="--", label="pred Ψ")
            a.plot(d["tau"], d["e"], color=AM, lw=2.4, label="true Ẽ")
            a.plot(d["tau"], r["e"], color=AM, lw=2, ls="--", label="pred Ẽ")
            a.set_xlabel("τ"); a.legend(frameon=False, fontsize=8); ax0(a)
            a.set_title("Shared Ψ & Ẽ recovery"); st.pyplot(f)
        with c2:
            f, a = plt.subplots(figsize=(6, 3.6), facecolor="none")
            for i, c in enumerate((CY, AM)):
                ct, cp = comb(d["K_true"][i], d["runs"][i]["h"], d["e"]), comb(r["Ks"][i], r["hs"][i], r["e"])
                a.plot(d["tau"], ct, color=c, lw=2.4, label=f"true run{i}")
                a.plot(d["tau"], cp, color=c, lw=2, ls="--", label=f"pred run{i}")
            a.set_xlabel("τ"); a.set_ylabel("K̃ĥ³+Ẽ"); a.legend(frameon=False, fontsize=8); ax0(a)
            a.set_title("Combined ODE term (the identifiable part)"); st.pyplot(f)
        st.caption("⚠️ Identifiability: K̃ and Ẽ trade off against each other — the *combined* term K̃ĥ³+Ẽ is what the data actually pins down.")
    else:
        st.info("Generate data + train first.")

with tb[4]:
    st.markdown("#### The model")
    st.latex(r"\frac{d\tilde h}{d\tau} \;=\; -\,\tilde K(\tau)\,\tilde h^{3} \;-\; \tilde E(\tau),"
             r"\qquad \tilde K(\tau)=\Big(\tfrac{\omega}{\omega_{\mathrm{ref}}}\Big)^{2}\Psi(\tau)")
    st.markdown("""
    - **ĥ(τ)** — dimensionless film thickness, **Ψ(τ)** shared latent convective term, **Ẽ(τ)** evaporation.
    - Two runs at different RPM share **Ψ** and **Ẽ**; only the ω² scaling of K̃ differs → this is what makes the
      inverse problem *identifiable in principle*.
    - The PINN enforces the ODE at unlabeled collocation points (physics loss) while fitting sparse noisy
      thickness data (data loss).
    - Known failure mode from the notebook: Ψ and Ẽ are individually hard to disentangle — the optimizer can
      trade one against the other while still matching h(τ) almost perfectly. Watch the **Results** tab.
    """)
