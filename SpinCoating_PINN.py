# =====================================================================
#  THIN-FILM INVERSE LAB  ·  spin coating + blade coating + manual data
#  One file. CPU-only.  Deps: streamlit numpy scipy torch matplotlib
#  Run dark: see .streamlit/config.toml or the --theme flags above.
# =====================================================================
import io, csv, math
import numpy as np
import streamlit as st
import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except Exception:  # older scipy
    from scipy.integrate import cumtrapz

st.set_page_config(page_title="Thin-Film Inverse Lab", page_icon="◈", layout="wide")

# ---------------------------------------------------------------------
#  DESIGN LAYER  ·  fonts + ambient background + flair (widgets already
#  themed dark by config; here we add texture, type contrast, motion)
# ---------------------------------------------------------------------
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,500;0,9..144,700;0,9..144,900;1,9..144,500&family=IBM+Plex+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
:root{--ink:#e8ecf4;--mute:#8d9bb2;--line:rgba(232,236,244,.10);
 --c:#34d6cf;--a:#f4b740;--d:#ff6f6b;--field:rgba(255,255,255,.04);}
html,body{font-family:'IBM Plex Sans',sans-serif;}
[data-testid="stAppViewContainer"]{background:
  radial-gradient(1100px 620px at 86% -10%, rgba(52,214,207,.10), transparent 60%),
  radial-gradient(900px 600px at -8% 108%, rgba(244,183,64,.08), transparent 60%),
  radial-gradient(700px 700px at 50% 120%, rgba(120,150,255,.05), transparent 60%),
  #0a0e16 !important;}
[data-testid="stAppViewContainer"]::before{content:"";position:fixed;inset:0;z-index:0;pointer-events:none;
  background-image:radial-gradient(rgba(232,236,244,.05) 1px,transparent 1px);background-size:26px 26px;}
[data-testid="stSidebar"]{background:rgba(13,18,26,.82)!important;border-right:1px solid var(--line);}
.stMarkdown,.stMarkdown p,.stMarkdown li{font-family:'IBM Plex Sans',sans-serif;color:var(--ink);}
.stMarkdown h1,.stMarkdown h2,.stMarkdown h3,[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{font-family:'Fraunces',serif;letter-spacing:-.01em;}
.grain{position:fixed;inset:0;z-index:1;pointer-events:none;opacity:.045;mix-blend-mode:overlay;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='2'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");}
.sheen{position:fixed;left:0;right:0;height:1px;z-index:1;pointer-events:none;opacity:.4;
  background:linear-gradient(90deg,transparent,var(--c),transparent);animation:sheen 9s linear infinite;}
@keyframes sheen{0%{top:-4%}100%{top:104%}}
@keyframes rise{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
.hero{animation:rise .7s both}
.kicker{font-family:'JetBrains Mono',monospace;font-size:.72rem;letter-spacing:.26em;
  text-transform:uppercase;color:var(--c);margin:0 0 14px;}
.title{font-family:'Fraunces',serif;font-weight:900;line-height:.96;
  font-size:clamp(2.5rem,5.6vw,4.6rem);margin:0;color:var(--ink);}
.title em{font-style:italic;font-weight:500;color:var(--a);}
.lede{max-width:64ch;color:var(--mute);font-weight:300;font-size:1.02rem;margin:14px 0 0;}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:16px;}
.chip{font-family:'JetBrains Mono',monospace;font-size:.72rem;padding:5px 11px;border-radius:999px;
  border:1px solid var(--line);color:var(--mute);background:var(--field);transition:.2s;}
.chip:hover{border-color:var(--c);color:var(--c);transform:translateY(-1px);}
.chip-c{border-color:rgba(52,214,207,.4);color:var(--c);}
.chip-a{border-color:rgba(244,183,64,.4);color:var(--a);}
.sec{margin:8px 0 18px;animation:rise .6s both;}
.sec-i{font-family:'JetBrains Mono',monospace;font-size:.7rem;letter-spacing:.2em;color:var(--a);}
.sec-t{font-family:'Fraunces',serif;font-weight:700;font-size:clamp(1.4rem,2.6vw,2rem);margin:2px 0 0;}
.sec-s{color:var(--mute);font-weight:300;margin:4px 0 0;max-width:70ch;}
.note{border-left:3px solid var(--c);background:rgba(52,214,207,.06);
  border-radius:0 10px 10px 0;padding:12px 16px;margin:14px 0;color:var(--ink);font-size:.92rem;}
.note b{color:var(--c);}
.note-a{border-left-color:var(--a);background:rgba(244,183,64,.06);}
.note-a b{color:var(--a);}
[data-testid="stMetric"]{background:linear-gradient(160deg,rgba(255,255,255,.05),rgba(255,255,255,.015));
  border:1px solid var(--line);border-radius:14px;padding:14px 16px;transition:.25s;}
[data-testid="stMetric"]:hover{transform:translateY(-3px);border-color:rgba(52,214,207,.45);
  box-shadow:0 14px 34px -20px rgba(52,214,207,.6);}
[data-testid="stMetricValue"]{font-family:'Fraunces',serif!important;font-weight:700;}
[data-testid="stMetricLabel"]{font-family:'JetBrains Mono',monospace!important;
  text-transform:uppercase;letter-spacing:.12em;font-size:.62rem!important;color:var(--mute)!important;}
.stTabs [data-baseweb="tab"]{font-family:'IBM Plex Sans',sans-serif!important;font-weight:600!important;}
.stTabs [data-baseweb="tab"]:hover{color:var(--c)!important;}
.stButton>button{border-radius:11px!important;font-family:'IBM Plex Sans',sans-serif!important;
  font-weight:600!important;transition:.2s!important;}
.stButton>button:hover{transform:translateY(-2px);}
[data-testid="stImage"]{transition:filter .3s;}
[data-testid="stImage"]:hover{filter:brightness(1.06);}
hr{border-color:var(--line)!important;}
</style>
<div class="grain"></div><div class="sheen"></div>
""", unsafe_allow_html=True)

plt.rcParams.update({"axes.facecolor":"none","figure.facecolor":"none",
 "axes.edgecolor":"#2a3340","axes.labelcolor":"#c4cedd","text.color":"#c4cedd",
 "xtick.color":"#8d9bb2","ytick.color":"#8d9bb2","axes.grid":True,
 "grid.color":"#1a212c","axes.spines.top":False,"axes.spines.right":False,
 "legend.frameon":False,"font.family":"IBM Plex Sans"})
CY, AM, RD, BL = "#34d6cf", "#f4b740", "#ff6f6b", "#7aa2ff"

# ---------------------------------------------------------------------
#  NETWORKS  (hard-IC thickness; positivity via exp / softplus)
# ---------------------------------------------------------------------
def mlp(h=24, L=2):
    lay = [nn.Linear(1, h), nn.Tanh()]
    for _ in range(L - 1):
        lay += [nn.Linear(h, h), nn.Tanh()]
    return nn.Sequential(*lay, nn.Linear(h, 1))

class HNet(nn.Module):                       # h(0)=h0 exactly; monotone-thinning ansatz
    def __init__(self):
        super().__init__(); self.net = mlp(); self.sp = nn.Softplus()
    def forward(self, t, h0=1.0):
        return h0 - t * self.sp(self.net(t))

class ENet(nn.Module):                       # evaporation >= 0
    def __init__(self):
        super().__init__(); self.net = mlp(); self.sp = nn.Softplus()
    def forward(self, t):
        return self.sp(self.net(t))

class PsiPar(nn.Module):                     # constrained Psi = A*exp(-d*tau), d>=0
    def __init__(self):
        super().__init__(); self.logA = nn.Parameter(torch.tensor(0.0))
        self.raw = nn.Parameter(torch.tensor(0.5)); self.sp = nn.Softplus()
    def forward(self, t):
        return torch.exp(self.logA - self.sp(self.raw) * t)
    def ab(self):
        return float(torch.exp(self.logA).item()), float(self.sp(self.raw).item())

class PsiFree(nn.Module):                    # unconstrained (mirror-image diagnostic)
    def __init__(self):
        super().__init__(); self.net = mlp()
    def forward(self, t):
        return torch.exp(self.net(t))

def _dh(hn, t):
    t = t.reshape(-1, 1); h = hn(t, 1.0)
    dh = torch.autograd.grad(h, t, grad_outputs=torch.ones_like(h),
                             create_graph=True, retain_graph=True)[0]
    return h, dh

def res_spin(hn, psi, en, t, w):
    h, dh = _dh(hn, t); return dh + (w ** 2) * psi(t) * h ** 3 + en(t)

def res_blade(hn, en, t):
    h, dh = _dh(hn, t); return dh + en(t)

def rel(p, t):
    p = np.asarray(p, float).ravel(); t = np.asarray(t, float).ravel()
    return float(np.mean(np.abs(p - t) / (np.abs(t) + 1e-8)) * 100)

def _tick(prog, status, ep, n, msg):
    if prog is not None:
        prog.progress((ep + 1) / n)
    if status is not None and ep % 25 == 0:
        status.caption(msg)

# ---------------------------------------------------------------------
#  TRAINERS
# ---------------------------------------------------------------------
def train_spin(runs, param_psi=True, free_diag=False, prog=None, status=None):
    W = [r["w"] for r in runs]
    td = [r["td"] for r in runs]; hd = [r["hd"] for r in runs]
    hn = [HNet() for _ in W]
    # phase A: fit h to data
    oA = optim.Adam([p for h in hn for p in h.parameters()], lr=1e-3)
    for ep in range(350):
        oA.zero_grad(); L = sum(torch.mean((hn[i](td[i], 1.0) - hd[i]) ** 2) for i in range(len(W)))
        L.backward(); oA.step(); _tick(prog, status, ep, 350, f"phase A · fit h · {L.item():.4f}")
    out = dict(hn=hn, spread=None, free=None)
    if param_psi:
        for h in hn:
            for p in h.parameters(): p.requires_grad_(False)
        psi, en = PsiPar(), ENet()
        oB = optim.Adam([{"params": psi.parameters(), "lr": 1e-2},
                         {"params": en.parameters(), "lr": 1e-3}])
        for ep in range(400):
            oB.zero_grad(); L = sum(torch.mean(res_spin(hn[i], psi, en,
                          torch.tensor(np.sort(np.random.uniform(0, 1, 120)),
                          dtype=torch.float32).reshape(-1, 1).requires_grad_(True), W[i]) ** 2)
                          for i in range(len(W)))
            L.backward(); oB.step(); _tick(prog, status, ep, 400, f"phase B · physics · {L.item():.4f}")
        for h in hn:
            for p in h.parameters(): p.requires_grad_(True)
        oC = optim.Adam([{"params": [p for h in hn for p in h.parameters()], "lr": 1e-4},
                         {"params": psi.parameters(), "lr": 1e-3},
                         {"params": en.parameters(), "lr": 1e-4}])
        for ep in range(250):
            oC.zero_grad(); Ld = Lp = 0.0
            for i in range(len(W)):
                Ld += torch.mean((hn[i](td[i], 1.0) - hd[i]) ** 2)
                tc = torch.tensor(np.sort(np.random.uniform(0, 1, 120)),
                                  dtype=torch.float32).reshape(-1, 1).requires_grad_(True)
                Lp += torch.mean(res_spin(hn[i], psi, en, tc, W[i]) ** 2)
            (Ld + Lp).backward(); oC.step()
            _tick(prog, status, ep, 250, f"phase C · joint · {Ld.item():.4f}")
        out["psi"], out["en"] = psi, en
        # multi-start -> practical-identifiability spread on (A, d)
        spread = []
        for s in range(3):
            torch.manual_seed(1000 + s); p2, e2 = PsiPar(), ENet()
            with torch.no_grad():
                p2.logA.copy_(torch.tensor(float(np.random.uniform(-1, 1))))
                p2.raw.copy_(torch.tensor(float(np.random.uniform(-1, 2))))
            o = optim.Adam([{"params": p2.parameters(), "lr": 1e-2},
                            {"params": e2.parameters(), "lr": 1e-3}])
            for ep in range(250):
                o.zero_grad(); L = sum(torch.mean(res_spin(hn[i], p2, e2,
                              torch.tensor(np.sort(np.random.uniform(0, 1, 120)),
                              dtype=torch.float32).reshape(-1, 1).requires_grad_(True), W[i]) ** 2)
                              for i in range(len(W)))
                L.backward(); o.step()
            spread.append(p2.ab())
        out["spread"] = spread
    if free_diag:
        pf, ef = PsiFree(), ENet()
        params = [p for h in hn for p in h.parameters()] + list(pf.parameters()) + list(ef.parameters())
        oF = optim.Adam(params, lr=1e-3)
        for ep in range(700):
            oF.zero_grad(); Ld = Lp = 0.0
            for i in range(len(W)):
                Ld += torch.mean((hn[i](td[i], 1.0) - hd[i]) ** 2)
                tc = torch.tensor(np.sort(np.random.uniform(0, 1, 120)),
                                  dtype=torch.float32).reshape(-1, 1).requires_grad_(True)
                Lp += torch.mean(res_spin(hn[i], pf, ef, tc, W[i]) ** 2)
            (Ld + Lp).backward(); oF.step()
        out["free"] = (pf, ef)
    return out

def train_blade(run, prog=None, status=None):
    td, hd = run["td"], run["hd"]
    hn, en = HNet(), ENet()
    o = optim.Adam([p for p in hn.parameters()] + list(en.parameters()), lr=1e-3)
    for ep in range(500):
        o.zero_grad()
        Ld = torch.mean((hn(td, 1.0) - hd) ** 2)
        tc = torch.tensor(np.sort(np.random.uniform(0, 1, 150)),
                          dtype=torch.float32).reshape(-1, 1).requires_grad_(True)
        Lp = torch.mean(res_blade(hn, en, tc) ** 2)
        (Ld + Lp).backward(); o.step()
        _tick(prog, status, ep, 500, f"blade PINN · {Ld.item():.4f} / {Lp.item():.4f}")
    # parametric evaporation: h(t)=1-(E0/Ed)(1-exp(-Ed t)) fit to data
    td_n = td.numpy().ravel(); hd_n = hd.numpy().ravel()
    def obj(x):
        E0, Ed = float(np.exp(x[0])), float(np.exp(x[1]))
        hmod = 1.0 - (E0 / Ed) * (1.0 - np.exp(-Ed * td_n))
        return float(np.mean((hmod - hd_n) ** 2))
    r = minimize(obj, [0.0, 0.0], method="Nelder-Mead",
                 options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 4000})
    E0p, Edp = float(np.exp(r.x[0])), float(np.exp(r.x[1]))
    return dict(hn=hn, en=en, E0=E0p, Ed=Edp)

# ---------------------------------------------------------------------
#  SYNTHETIC BUILDERS
# ---------------------------------------------------------------------
def build_spin_syn(PA, PD, EB, ED, rpm_a, rpm_b, n_meas, noise, seed):
    np.random.seed(seed); torch.manual_seed(seed)
    W = [rpm_a / rpm_a, rpm_b / rpm_a]; rpms = [rpm_a, rpm_b]
    Psi = lambda t: PA * np.exp(-PD * t); E = lambda t: EB * np.exp(-ED * t)
    te = np.linspace(0, 1, 300); runs = []
    for w in W:
        s = solve_ivp(lambda t, h, w=w: [-(w ** 2) * Psi(t) * h[0] ** 3 - E(t)],
                      (0, 1), [1.0], t_eval=te, method="RK45")
        edges = np.linspace(0, 1, n_meas + 1)
        tg = np.array([np.random.uniform(edges[k], edges[k + 1]) for k in range(n_meas)])
        idx = np.sort(np.unique([np.argmin(np.abs(te - t)) for t in tg])); idx[-1] = len(te) - 1
        ht = s.y[0][idx]; hm = np.clip(ht + noise * ht * np.random.normal(0, 1, len(idx)), 1e-4, None)
        runs.append(dict(td=torch.tensor(te[idx], dtype=torch.float32).reshape(-1, 1),
                         hd=torch.tensor(hm, dtype=torch.float32).reshape(-1, 1), w=w,
                         h_true=s.y[0]))
    meta = dict(te=te, Psi=Psi(te), E=E(te), rpms=rpms, true=True, mode="synthetic")
    return runs, meta

def build_blade_syn(E0, Ed, h_wet, t_ref, U, sigma, H0, n_meas, noise, seed):
    np.random.seed(seed); torch.manual_seed(seed)
    te = np.linspace(0, 1, 300); E = E0 * np.exp(-Ed * te)
    ht = 1.0 - cumtrapz(E, te, initial=0.0); ht = np.clip(ht, 1e-3, None)
    edges = np.linspace(0, 1, n_meas + 1)
    tg = np.array([np.random.uniform(edges[k], edges[k + 1]) for k in range(n_meas)])
    idx = np.sort(np.unique([np.argmin(np.abs(te - t)) for t in tg])); idx[-1] = len(te) - 1
    hm = np.clip(ht[idx] + noise * ht[idx] * np.random.normal(0, 1, len(idx)), 1e-3, None)
    run = dict(td=torch.tensor(te[idx], dtype=torch.float32).reshape(-1, 1),
               hd=torch.tensor(hm, dtype=torch.float32).reshape(-1, 1))
    mu_true = sigma / U * (h_wet / H0) ** 1.5
    meta = dict(te=te, E=E, ht=ht, h_wet=h_wet, t_ref=t_ref, U=U, sigma=sigma,
                H0=H0, mu_true=mu_true, true=True, mode="synthetic")
    return run, meta

# ---------------------------------------------------------------------
#  MANUAL PARSER  (textarea CSV  OR  editable spreadsheet)
# ---------------------------------------------------------------------
_EXAMPLE = """run_id,t,h,rpm
0,0,1.000,3000
0,10,0.430,3000
0,20,0.180,3000
0,30,0.084,3000
1,0,1.000,4500
1,10,0.250,4500
1,20,0.070,4500
1,30,0.012,4500"""

def _norm_header(row):
    return {k.strip().lower().lstrip("﻿"): v for k, v in row.items()}

def parse_manual(text, df, h_wet, t_ref):
    rows = []
    if text.strip():
        rd = csv.DictReader(io.StringIO(text))
        for row in rd:
            row = _norm_header(row)
            try:
                t = float(row["t"]); h = float(row["h"])
            except (KeyError, ValueError, TypeError):
                continue
            rid = int(float(row["run_id"])) if row.get("run_id", "").strip() not in ("", None) else 0
            rpm = float(row["rpm"]) if row.get("rpm", "").strip() not in ("", None) else None
            rows.append((rid, t, h, rpm))
    elif df is not None and len(df):
        cols = [c.lower() for c in df.columns]
        for r in df.itertuples(index=False):
            d = dict(zip(cols, r))
            try:
                t = float(d["t"]); h = float(d["h"])
                if math.isnan(t) or math.isnan(h):
                    continue
            except (KeyError, ValueError, TypeError):
                continue
            rid = int(d["run_id"]) if "run_id" in d and not (isinstance(d["run_id"], float) and math.isnan(d["run_id"])) else 0
            rpm = d.get("rpm")
            rpm = None if (rpm is None or (isinstance(rpm, float) and math.isnan(rpm))) else float(rpm)
            rows.append((rid, t, h, rpm))
    if not rows:
        return None, "no readable rows (need at least columns t, h)"
    if h_wet <= 0 or t_ref <= 0:
        return None, "h_wet and t_ref must be > 0"
    groups = {}
    for rid, t, h, rpm in rows:
        groups.setdefault(rid, dict(t=[], h=[], rpm=rpm))
        groups[rid]["t"].append(t); groups[rid]["h"].append(h)
        if rpm is not None:
            groups[rid]["rpm"] = rpm
    runs = []
    for rid in sorted(groups):
        o = np.argsort(groups[rid]["t"])
        t_raw = np.array(groups[rid]["t"])[o]; h_raw = np.array(groups[rid]["h"])[o]
        if len(t_raw) < 2:
            return None, f"run {rid} has < 2 points"
        runs.append(dict(td=torch.tensor(t_raw / t_ref, dtype=torch.float32).reshape(-1, 1),
                         hd=torch.tensor(h_raw / h_wet, dtype=torch.float32).reshape(-1, 1),
                         rpm=groups[rid]["rpm"], t_raw=t_raw, h_raw=h_raw))
    return dict(runs=runs, h_wet=h_wet, t_ref=t_ref), None

# ---------------------------------------------------------------------
#  UI HELPERS
# ---------------------------------------------------------------------
def sec(i, t, s=""):
    st.markdown(f'<div class="sec"><div class="sec-i">{i}</div>'
                f'<div class="sec-t">{t}</div><div class="sec-s">{s}</div></div>',
                unsafe_allow_html=True)

def note(html, amber=False):
    st.markdown(f'<div class="note{"-a" if amber else ""}">{html}</div>', unsafe_allow_html=True)

def fig():
    f, a = plt.subplots(figsize=(6.4, 3.7)); return f, a

# ---------------------------------------------------------------------
#  HERO
# ---------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <div class="kicker">physics-informed inverse lab · thin-film drying</div>
  <h1 class="title">Read the film.<br><em>Infer</em> the physics.</h1>
  <p class="lede">A PINN watches a thinning film and tries to recover the hidden
  viscosity &amp; evaporation that caused it. Two coating processes, synthetic or
  <b>your own</b> thickness data, with the identifiability limits shown honestly
  instead of hidden.</p>
  <div class="chips">
    <span class="chip chip-c">dh̃/dτ = −w²Ψh³ − Ẽ</span>
    <span class="chip chip-a">blade: dh̃/dτ = −Ẽ</span>
    <span class="chip">manual CSV / spreadsheet</span>
    <span class="chip">trust verdict per run</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
#  SIDEBAR
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="kicker" style="margin-top:6px">controls</div>', unsafe_allow_html=True)
    source = st.radio("data source", ["Synthetic", "Manual (Manual-input tab)"], index=0)
    is_manual = source.startswith("Manual")

    with st.expander("nondimensionalise manual data", expanded=is_manual):
        st.caption("τ = t / t_ref ,  h̃ = h / h_wet.  If your data is already "
                   "dimensionless (h starts at 1, τ in [0,1]) set both to 1.")
        h_wet = st.number_input("h_wet (same units as h)", 1e-6, 1e6, 1.0, format="%.4f")
        t_ref = st.number_input("t_ref (same units as t)", 1e-6, 1e6, 1.0, format="%.4f")

    with st.expander("synthetic · spin physics", expanded=(not is_manual)):
        PA = st.slider("Ψ_A", 0.1, 3.0, 1.2, 0.05)
        PD = st.slider("Ψ decay", 0.5, 6.0, 3.0, 0.1)
        EB = st.slider("Ẽ_B", 0.5, 6.0, 3.0, 0.1)
        ED = st.slider("Ẽ decay", 0.5, 6.0, 3.5, 0.1)
        rpm_a = st.slider("run A · RPM", 1000, 6000, 3000, 100)
        rpm_b = st.slider("run B · RPM", 1000, 6000, 4500, 100)
        n_meas = st.slider("measurements / run", 4, 24, 8)
        noise = st.slider("noise σ", 0.0, 0.1, 0.02, 0.005)
        seed = st.number_input("seed", 0, 999, 42)

    with st.expander("synthetic · blade physics", expanded=False):
        bE0 = st.slider("Ẽ_0", 0.5, 6.0, 3.0, 0.1)
        bEd = st.slider("Ẽ decay", 0.5, 6.0, 3.5, 0.1)
        bhw = st.number_input("h_wet (m)", 1e-7, 1e-3, 2e-6, format="%.1e")
        btref = st.number_input("t_ref (s)", 1e-2, 1e2, 30.0, format="%.1f")
        bU = st.number_input("substrate speed U (m/s)", 1e-3, 1.0, 0.05, format="%.3f")
        bsg = st.number_input("surface tension σ (N/m)", 1e-3, 1e-1, 0.03, format="%.3f")
        bH0 = st.number_input("LL constant H0 (m)", 1e-5, 1e-2, 1e-3, format="%.1e")
        bseed = st.number_input("seed", 0, 999, 7, key="bseed")

    st.markdown("---")
    st.caption("◈  Reduced blade model = Landau–Levich entrainment + evaporative "
               "drying, used as a synthetic contrast. Not a validated full blade solver.")

# ---------------------------------------------------------------------
#  TABS
# ---------------------------------------------------------------------
t_spin, t_blade, t_manual, t_model = st.tabs(
    ["◐  Spin coating", "◑  Blade coating", "✎  Manual input", "◈  Model & caveats"])

# ============================ SPIN ====================================
with t_spin:
    sec("01", "Spin-coating inverse",
        "Per-run thickness nets, a <b>shared</b> Ψ(τ) and Ẽ(τ). Two spin speeds give the "
        "ω² lever that breaks the Ψ/Ẽ degeneracy — in principle.")
    c1, c2 = st.columns([1, 2])
    with c1:
        param_psi = st.checkbox("constrained Ψ = A·e^(−dτ)", True)
        free_diag = st.checkbox("also show free-Ψ diagnostic", False)
        run_spin = st.button("▶  Train inverse model", use_container_width=True, key="runspin")
        prog = st.progress(0); status = st.empty()
    with c2:
        if is_manual:
            m = st.session_state.get("manual")
            if m is None:
                st.warning("No manual data parsed yet — open the **Manual input** tab.")
                runs = meta = None
            else:
                miss = [i for i, r in enumerate(m["runs"]) if r.get("rpm") is None]
                if miss:
                    st.error(f"Spin needs an RPM per run; run(s) {miss} lack it.")
                    runs = meta = None
                else:
                    rpm_ref = min(r["rpm"] for r in m["runs"])
                    runs = [dict(td=r["td"], hd=r["hd"], w=r["rpm"] / rpm_ref) for r in m["runs"]]
                    meta = dict(rpms=[r["rpm"] for r in m["runs"]], true=False, mode="manual")
                    st.caption(f"manual · {len(runs)} run(s), rpms {meta['rpms']}")
        else:
            runs, meta = build_spin_syn(PA, PD, EB, ED, rpm_a, rpm_b, n_meas, noise, seed)
        sig = ("spin", source, str(runs is not None),
               "" if is_manual else f"{PA}{PD}{EB}{ED}{rpm_a}{rpm_b}{n_meas}{noise}{seed}{param_psi}{free_diag}")

        if run_spin and runs is not None:
            res = train_spin(runs, param_psi=param_psi, free_diag=free_diag,
                             prog=prog, status=status)
            res["sig"] = sig
            st.session_state["spin_res"] = res
            prog.progress(1.0); status.caption("done ✓")
        elif run_spin and runs is None:
            st.stop()

        res = st.session_state.get("spin_res")
        if res and res.get("sig") != sig:
            st.info("Parameters changed — click **▶ Train** to refresh.")

        if res:
            hn = res["hn"]; te = np.linspace(0, 1, 300)
            tt = torch.tensor(te, dtype=torch.float32).reshape(-1, 1)
            with torch.no_grad():
                Pp = res["psi"](tt).numpy().ravel() if param_psi else res["free"][0](tt).numpy().ravel()
                Ep = res["en"](tt).numpy().ravel()
                hp = [hn[i](tt, 1.0).numpy().ravel() for i in range(len(runs))]
            m1, m2, m3, m4 = st.columns(4)
            if meta["true"]:
                m1.metric("Ψ error", f"{rel(Pp, meta['Psi']):.0f}%")
                m2.metric("Ẽ error", f"{rel(Ep, meta['E']):.0f}%")
            else:
                m1.metric("Ψ (no truth)", "—"); m2.metric("Ẽ (no truth)", "—")
            m3.metric("h̄ error", f"{np.mean([rel(hp[i], runs[i]['hd'].numpy() if not meta['true'] else runs[i]['h_true']) for i in range(len(runs))]):.1f}%")
            # combined term (the identifiable quantity)
            if meta["true"]:
                ce = []
                for i in range(len(runs)):
                    w = runs[i]["w"]
                    ct = (w ** 2) * meta["Psi"] * runs[i]["h_true"] ** 3 + meta["E"]
                    cp = (w ** 2) * Pp * hp[i] ** 3 + Ep
                    ce.append(rel(cp, ct))
                m4.metric("combined", f"{np.mean(ce):.0f}%")
            else:
                m4.metric("combined", "—")

            # trust verdict from multi-start spread + horizon
            if res.get("spread"):
                A = np.array([a for a, _ in res["spread"]]); d = np.array([b for _, b in res["spread"]])
                arel = np.std(A) / max(abs(np.median(A)), 1e-6)
                drel = np.std(d) / max(abs(np.median(d)), 1e-6)
                vA = "ok-ish" if arel < 0.3 else "poorly pinned"
                vD = "UNIDENTIFIABLE" if drel > 0.5 else "marginal"
            else:
                arel = drel = None; vA = vD = "n/a (free Ψ)"
            # information horizon (needs >=2 runs)
            hor = None
            if len(runs) >= 2:
                with torch.no_grad():
                    hh = [hn[i](tt, 1.0).numpy().ravel() for i in (0, 1)]
                c = np.abs((runs[0]["w"] ** 2) * hh[0] ** 3 - (runs[1]["w"] ** 2) * hh[1] ** 3)
                tot = np.trapz(c, te); e = te < 0.2
                hor = float(np.trapz(c[e], te[e]) / tot) if tot > 0 else float("nan")

            note(f"<b>Trust verdict.</b> Ψ amplitude: <b>{vA}</b> · Ψ decay: <b>{vD}</b>"
                 + (f" · multi-start spread/median A={arel:.2f}, d={drel:.2f}." if arel is not None else ".")
                 + (" · 2-run ω² lever present." if len(runs) >= 2 else " · single run → no ω² lever."))
            if hor is not None:
                note(f"<b>Information horizon.</b> {100*hor:.0f}% of the 2-run Ψ leverage sits in "
                     "τ&lt;0.2 — past that the film is too thin to feel viscosity. The Ψ <b>decay</b> "
                     "lives exactly there, so it is <b>not in the data</b> (the noise sweep is flat).",
                     amber=True)

            fa, faa = fig()
            if meta["true"]:
                faa.plot(te, meta["Psi"], c=CY, lw=2.4, label="true Ψ")
                faa.plot(te, meta["E"], c=AM, lw=2.4, label="true Ẽ")
            faa.plot(te, Pp, "--", c=CY, lw=2, label="pred Ψ")
            faa.plot(te, Ep, "--", c=AM, lw=2, label="pred Ẽ")
            faa.set_title("shared Ψ & "); faa.legend(fontsize=8); fa.tight_layout(); st.pyplot(fa)

            fb, fbb = fig()
            for i in range(len(runs)):
                fbb.plot(te if meta["true"] else te, runs[i]["h_true"] if meta["true"] else hp[i],
                         c=[CY, AM][i % 2], alpha=.4)
                fbb.plot(te, hp[i], "--", c=[CY, AM][i % 2])
                fbb.scatter(runs[i]["td"].numpy(), runs[i]["hd"].numpy(), c=[CY, AM][i % 2], s=22)
            fbb.set_title("thickness (data + fit)"); fb.tight_layout(); st.pyplot(fb)

            if res.get("free"):
                with torch.no_grad():
                    Pf = res["free"][0](tt).numpy().ravel()
                fc, fcc = fig()
                if meta["true"]:
                    fcc.plot(te, meta["Psi"], c=CY, lw=2.4, label="true Ψ")
                fcc.plot(te, Pf, "--", c=RD, lw=2, label="free Ψ (mirror image)")
                fcc.set_title("free-Ψ diagnostic — fits the sum, hallucinates the split")
                fcc.legend(fontsize=8); fc.tight_layout(); st.pyplot(fc)

            if hor is not None:
                hd, hdd = fig()
                hdd.plot(te, c, c="#c4cedd", lw=2)
                hdd.axvline(0.2, c=RD, ls=":")
                hdd.set_title(f"2-run leverage c(τ) · {100*hor:.0f}% in τ<0.2")
                hd.tight_layout(); st.pyplot(hd)
        else:
            st.info("Configure the data source, then click **▶ Train**.")

# ============================ BLADE ===================================
with t_blade:
    sec("02", "Blade-coating inverse (reduced model)",
        "Wet thickness set by Landau–Levich entrainment h_w=H₀(μU/σ)^⅔, then <b>evaporative "
        "drying only</b>: dh̃/dτ=−Ẽ. Here Ẽ is the thinning rate itself — so it is directly "
        "observable, in sharp contrast to spin coating.")
    c1, c2 = st.columns([1, 2])
    with c1:
        run_blade = st.button("▶  Train inverse model", use_container_width=True, key="runblade")
        prog2 = st.progress(0); status2 = st.empty()
    with c2:
        if is_manual:
            m = st.session_state.get("manual")
            if m is None:
                st.warning("No manual data parsed yet — open the **Manual input** tab.")
                brun = bmeta = None
            else:
                brun = m["runs"][0]; bmeta = dict(h_wet=m["h_wet"], t_ref=m["t_ref"],
                          U=bU, sigma=bsg, H0=bH0, true=False, mode="manual")
                st.caption(f"manual · run 0 · {len(brun['td'])} pts")
        else:
            brun, bmeta = build_blade_syn(bE0, bEd, bhw, btref, bU, bsg, bH0, n_meas, noise, bseed)
        sigb = ("blade", source, str(brun is not None),
                "" if is_manual else f"{bE0}{bEd}{bhw}{btref}{bU}{bsg}{bH0}{n_meas}{noise}{bseed}")

        if run_blade and brun is not None:
            bres = train_blade(brun, prog=prog2, status=status2)
            bres["sig"] = sigb; st.session_state["blade_res"] = bres
            prog2.progress(1.0); status2.caption("done ✓")
        elif run_blade and brun is None:
            st.stop()
        bres = st.session_state.get("blade_res")
        if bres and bres.get("sig") != sigb:
            st.info("Parameters changed — click **▶ Train** to refresh.")

        if bres:
            te = np.linspace(0, 1, 300); tt = torch.tensor(te, dtype=torch.float32).reshape(-1, 1)
            with torch.no_grad():
                Ep = bres["en"](tt).numpy().ravel()
                hp = bres["hn"](tt, 1.0).numpy().ravel()
            E0p, Edp = bres["E0"], bres["Ed"]
            ht_param = 1.0 - (E0p / Edp) * (1.0 - np.exp(-Edp * te))
            m1, m2, m3 = st.columns(3)
            if bmeta["true"]:
                m1.metric("Ẽ error (PINN)", f"{rel(Ep, bmeta['E']):.0f}%")
                m2.metric("Ẽ error (param)", f"{rel(1.0-(E0p/Edp)*(1-np.exp(-Edp*te)), bmeta['ht']):.0f}%")
                m3.metric("h̃ error", f"{rel(hp, bmeta['ht']):.1f}%")
            else:
                m1.metric("Ẽ (no truth)", "—"); m2.metric("param Ẽ", f"E0={E0p:.2f} d={Edp:.2f}")
                m3.metric("h̃ error", f"{rel(hp, brun['hd'].numpy()):.1f}%")

            note("<b>Why blade differs from spin.</b> With no convective term competing, "
                 "Ẽ(τ) = −dh̃/dτ is read almost straight off the thinning curve — so evaporation "
                 "is <b>well identified</b> here, the opposite of spin coating where Ẽ and Ψh³ tangle.")

            # mu identifiability demo: mu only enters via h_w (the IC), NOT the curve
            hw_used = bmeta["h_wet"]
            hw_grid = np.linspace(0.7, 1.3, 60) * hw_used
            mu_grid = bmeta["sigma"] / bmeta["U"] * (hw_grid / bmeta["H0"]) ** 1.5
            note("<b>Viscosity is NOT in the drying curve.</b> μ enters only through the "
                 "measured wet thickness h_w (the initial condition). Mis-measure h_w by 10% and "
                 "the μ estimate shifts by ~15% while the Ẽ fit is <b>unchanged</b> — the curve "
                 "carries zero information about μ. (Reduced-model scope: a full Bornside / "
                 "viscocapillary model could in principle couple them; that is out of scope here.)",
                 amber=True)

            fa, faa = fig()
            if bmeta["true"]:
                faa.plot(te, bmeta["ht"], c=CY, lw=2.4, label="true h̃")
            faa.plot(te, hp, "--", c=AM, lw=2, label="PINN h̃")
            faa.plot(te, ht_param, ":", c=BL, lw=2, label="param h̃")
            faa.scatter(brun["td"].numpy(), brun["hd"].numpy(), c="#c4cedd", s=22, label="data")
            faa.set_title("drying curve"); faa.legend(fontsize=7); fa.tight_layout(); st.pyplot(fa)

            fb, fbb = fig()
            if bmeta["true"]:
                fbb.plot(te, bmeta["E"], c=CY, lw=2.4, label="true Ẽ")
            fbb.plot(te, Ep, "--", c=AM, lw=2, label="PINN Ẽ")
            fbb.plot(te, E0p * np.exp(-Edp * te), ":", c=BL, lw=2, label="param Ẽ")
            fbb.set_title("evaporation = thinning rate"); fbb.legend(fontsize=7)
            fb.tight_layout(); st.pyplot(fb)

            fc, fcc = fig()
            fcc.plot(hw_grid * 1e6, mu_grid, c=AM, lw=2)
            fcc.axvline(hw_used * 1e6, c=CY, ls=":", label=f"measured h_w={hw_used*1e6:.2f}µm")
            if bmeta["true"]:
                fcc.axhline(bmeta["mu_true"], c=RD, ls="--", alpha=.6, label="true μ")
            fcc.set_xlabel("assumed h_w (µm)"); fcc.set_ylabel("μ estimate (Pa·s)")
            fcc.set_title("μ vs h_w — flat curve fit, moving μ"); fcc.legend(fontsize=7)
            fc.tight_layout(); st.pyplot(fc)
        else:
            st.info("Configure the data source, then click **▶ Train**.")

# ============================ MANUAL ==================================
with t_manual:
    sec("03", "Manual input",
        "Paste a CSV or type into the spreadsheet. Columns: <code>t, h</code> required; "
        "<code>run_id</code> and <code>rpm</code> optional (rpm needed for spin). Values are "
        "raw; set h_wet &amp; t_ref in the sidebar to nondimensionalise.")
    ca, cb = st.columns(2)
    with ca:
        st.markdown("**paste CSV**")
        txt = st.text_area(" ", value="", height=220, key="csvtxt",
                           placeholder=_EXAMPLE)
        if st.button("load example", key="ldex"):
            st.session_state["csvtxt"] = _EXAMPLE
            st.rerun()
    with cb:
        st.markdown("**or type / edit**")
        try:
            df0 = st.session_state.get("df0")
            if df0 is None:
                df0 = __import__("pandas").DataFrame(
                    columns=["run_id", "t", "h", "rpm"],
                    index=range(5)).astype({"run_id": "float", "t": "float",
                                            "h": "float", "rpm": "float"})
            df = st.data_editor(df0, num_rows="dynamic", use_container_width=True,
                                key="dfedit", hide_index=True)
            st.session_state["df0"] = df
        except Exception as ex:
            st.caption("spreadsheet unavailable here; use the CSV box.")
            df = None

    if st.button("parse & store", use_container_width=True, type="primary", key="parse"):
        h_wet_v = h_wet if is_manual else st.session_state.get("h_wet_m", 1.0)
        t_ref_v = t_ref if is_manual else st.session_state.get("t_ref_v", 1.0)
        # when synthetic source, manual tab still needs h_wet/t_ref; fall back to 1.0
        if not is_manual:
            h_wet_v, t_ref_v = 1.0, 1.0
        parsed, err = parse_manual(st.session_state.get("csvtxt", ""), df, h_wet_v, t_ref_v)
        if err:
            st.error(err)
        else:
            st.session_state["manual"] = parsed
            rps = sorted({r["rpm"] for r in parsed["runs"] if r.get("rpm") is not None})
            st.success(f"stored {len(parsed['runs'])} run(s) · "
                       + (f"rpms {rps} · " if rps else "no rpm (ok for blade) · ")
                       + f"h_wet={parsed['h_wet']}, t_ref={parsed['t_ref']}")
    if st.session_state.get("manual"):
        m = st.session_state["manual"]
        st.caption(f"currently stored: {len(m['runs'])} run(s), "
                   f"{sum(len(r['td']) for r in m['runs'])} points, h_wet={m['h_wet']}, t_ref={m['t_ref']}")
        note("Switch <b>data source → Manual</b> in the sidebar, then Train in the "
             "Spin or Blade tab. Spin requires an <code>rpm</code> per run; blade uses run 0.",
             amber=True)

# ============================ MODEL ===================================
with t_model:
    sec("04", "Model & honest caveats",
        "What each model actually solves, and what it provably cannot.")
    st.markdown("""
**Spin coating.** Dimensionless thinning `dh̃/dτ = −w²Ψ(τ)h̃³ − Ẽ(τ)`. Per‑run thickness nets
carry a hard initial condition `h̃(0)=1`; Ψ and Ẽ are shared, with the ω² scaling providing the
only lever that separates them. With a constrained Ψ=A·e^(−dτ) the **amplitude** is recoverable to
the right order and the **shape** is correct, but the **decay rate is not** — and the app shows
*why*: the 2‑run leverage collapses after τ≈0.2 (the film is too thin to feel viscosity), which is
exactly where the decay lives. The flat noise‑sweep confirms this is informational, not a tuning
failure. The free‑Ψ diagnostic visualises the degeneracy directly (it fits the *sum*, hallucinates
the split).

**Blade coating (reduced).** Wet thickness from Landau–Levich entrainment, then evaporative drying
`dh̃/dτ = −Ẽ`. With no convective term competing, Ẽ is essentially the thinning rate and is
**well identified** — the clean contrast to spin. Viscosity μ enters only through the measured wet
thickness (the initial condition), so the drying curve carries **no** information about μ; the app
demonstrates this with the μ‑vs‑h_w plot.
""")
    note("<b>Scope, stated plainly.</b> The blade model here is the <b>reduced</b> entrainment + "
         "evaporation model used as a synthetic contrast — it is <b>not</b> a validated full "
         "blade‑coating inverse solver. A full Bornside / viscocapillary model could couple μ into "
         "the dynamics and is out of scope. Spin findings come from the project's own sandbox "
         "experiments (recalibrated K/Ẽ≈0.40).", amber=True)
    note("<b>Manual data.</b> Raw values are nondimensionalised by your h_wet &amp; t_ref. If you "
         "only have dimensionless traces, set both to 1 and treat any dimensional μ estimate as "
         "illustrative. Spin needs an RPM per run to form the ω² lever.")
    st.markdown("""
**Take‑home.** Thickness and evaporation are recoverable from sparse thickness‑vs‑time; the
viscosity *amplitude/shape* is recoverable under a physical constraint; the viscosity *decay* is
recoverable **only** when viscosity actually shapes the thinning (high K/Ẽ, or dense early
sampling, or a viscosity‑sensitive observable). The lab lever this gives is concrete: terminal‑
weighted sparse ellipsometry cannot see viscosity dynamics — sample the first ~15–20% of drying
densely, or measure concentration/rheology directly.
""")
