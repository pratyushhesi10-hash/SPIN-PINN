# app.py — SpinCoat PINN Lab  (Manual/CSV + honest verdict + consistency flag)
# Run:  streamlit run app.py
#
# NEW in this build:
#   (1) the Trust verdict reads the REAL per-run h-fit (no more "h: HIGH by construction");
#   (2) a model-consistency flag compares the joint-phase data-loss to the data-only
#       data-loss and raises a FLAG when the ODE cannot produce your traces.
# Your Manual / CSV tab is preserved; width/layers sliders are now actually wired.
import io, csv, math
import numpy as np
import pandas as pd
import streamlit as st
import torch, torch.nn as nn, torch.optim as optim
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="SpinCoat PINN Lab", page_icon="🌀", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
html,body,[class*="css"],.stMarkdown{font-family:'IBM Plex Sans',sans-serif;}
[data-testid="stAppViewContainer"]{background:
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
.note{border-left:3px solid #22d3ee;background:rgba(34,211,238,.06);border-radius:0 10px 10px 0;
  padding:12px 16px;margin:14px 0;color:#cbd5e1;font-size:.9rem;}
.note b{color:#22d3ee;} .note-a{border-left-color:#fbbf24;background:rgba(251,191,36,.06);}
.note-a b{color:#fbbf24;}
.status{border-left:3px solid #94a3b8;background:rgba(148,163,184,.06);border-radius:0 10px 10px 0;
  padding:12px 16px;margin:10px 0;}
.status-ok{border-left-color:#34d399;background:rgba(52,211,153,.08);}
.status-warn{border-left-color:#fbbf24;background:rgba(251,191,36,.08);}
.status-bad{border-left-color:#ff6b6b;background:rgba(255,107,107,.10);}
.status .st-t{font-family:'IBM Plex Mono',monospace;font-size:.7rem;letter-spacing:.14em;
  text-transform:uppercase;margin-bottom:8px;}
.status-ok .st-t{color:#34d399;} .status-warn .st-t{color:#fbbf24;} .status-bad .st-t{color:#ff6b6b;}
.status ul{margin:0;padding-left:18px;color:#cbd5e1;font-size:.85rem;line-height:1.55;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="hero-top"><span class="spin">🌀</span>
    <div><div class="kicker">Physics-Informed Neural Network · Inverse Discovery</div>
    <h1 class="title">SpinCoat <span class="accent">PINN</span> Lab</h1></div></div>
  <div class="chips">
    <span class="chip">dĥ/dτ = −w²Ψ(τ)ĥ³ − Ẽ(τ)</span>
    <span class="chip chip-cyan">shared Ψ &amp; Ẽ across runs</span>
    <span class="chip chip-amber">your data or synthetic</span>
    <span class="chip">honest trust verdict</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ============================== networks ==============================
def mlp(h=24, L=2):
    lay = [nn.Linear(1, h), nn.Tanh()]
    for _ in range(L - 1):
        lay += [nn.Linear(h, h), nn.Tanh()]
    lay += [nn.Linear(h, 1)]
    return nn.Sequential(*lay)

class HNet(nn.Module):
    def __init__(self, width=24, layers=2):
        super().__init__()
        self.net = mlp(width, layers)
        self.sp = nn.Softplus()
    def forward(self, t, h0=1.0):
        return h0 - t * self.sp(self.net(t))

class PsiPar(nn.Module):   # constrained Psi = A*exp(-d*tau), d>=0
    def __init__(self):
        super().__init__()
        self.logA = nn.Parameter(torch.tensor(0.0))
        self.raw = nn.Parameter(torch.tensor(0.5))
        self.sp = nn.Softplus()
    def forward(self, t):
        return torch.exp(self.logA - self.sp(self.raw) * t)
    def ab(self):
        return float(torch.exp(self.logA).item()), float(self.sp(self.raw).item())

class ENet(nn.Module):
    def __init__(self, width=24, layers=2):
        super().__init__()
        self.net = mlp(width, layers)
        self.sp = nn.Softplus()
    def forward(self, t):
        return self.sp(self.net(t))

class PsiFree(nn.Module):  # unconstrained diagnostic
    def __init__(self, width=24, layers=2):
        super().__init__()
        self.net = mlp(width, layers)
    def forward(self, t):
        return torch.exp(self.net(t))

def resid(hn, psi, en, t, w):
    t = t.reshape(-1, 1); h = hn(t, 1.0)
    dh = torch.autograd.grad(h, t, grad_outputs=torch.ones_like(h),
                             create_graph=True, retain_graph=True)[0]
    return dh + (w ** 2) * psi(t) * h ** 3 + en(t)

def coll(n):
    t = torch.tensor(np.sort(np.random.uniform(0, 1, n)), dtype=torch.float32).reshape(-1, 1)
    t.requires_grad_(True); return t

def set_grad(hn, on):
    for h in hn:
        for p in h.parameters(): p.requires_grad = on

# ---- training: now also returns the data-loss at end of Phase A and Phase C ----
def train_parametric(runs, ea, eb, ec, lr, wd, wp, width, layers, seed):
    torch.manual_seed(seed); W = [r['w'] for r in runs]
    td = [r['td'] for r in runs]; hd = [r['hd'] for r in runs]
    hn = [HNet(width, layers) for _ in W]; psi = PsiPar(); en = ENet(width, layers)
    # Phase A: data-only fit of h  (record its final data-loss = "best the data alone can do")
    oA = optim.Adam([p for h in hn for p in h.parameters()], lr=lr)
    Ld_A = 0.0
    for _ in range(ea):
        oA.zero_grad()
        L = sum(torch.mean((hn[i](td[i], 1.0) - hd[i]) ** 2) for i in range(len(W)))
        L.backward(); oA.step(); Ld_A = float(L.item())
    # Phase B: physics on frozen h
    for h in hn:
        for p in h.parameters(): p.requires_grad_(False)
    oB = optim.Adam([{'params': psi.parameters(), 'lr': lr * 10},
                     {'params': en.parameters(), 'lr': lr}])
    for _ in range(eb):
        oB.zero_grad()
        L = sum(torch.mean(resid(hn[i], psi, en, coll(150), W[i]) ** 2) for i in range(len(W)))
        L.backward(); oB.step()
    # Phase C: joint  (record its final data-loss = "data-loss once physics is enforced")
    for h in hn:
        for p in h.parameters(): p.requires_grad_(True)
    oC = optim.Adam([{'params': [p for h in hn for p in h.parameters()], 'lr': lr * 0.1},
                     {'params': psi.parameters(), 'lr': lr},
                     {'params': en.parameters(), 'lr': lr * 0.1}])
    Ld_C = 0.0
    for _ in range(ec):
        oC.zero_grad(); Ld = Lp = 0.0
        for i in range(len(W)):
            Ld = Ld + torch.mean((hn[i](td[i], 1.0) - hd[i]) ** 2)
            Lp = Lp + torch.mean(resid(hn[i], psi, en, coll(150), W[i]) ** 2)
        (wd * Ld + wp * Lp).backward(); oC.step(); Ld_C = float(Ld.item())
    return dict(hn=hn, psi=psi, en=en, Ld_A=Ld_A, Ld_C=Ld_C)

def multistart(hn, runs, ns, ep, lr, width=24, layers=2):
    W = [r['w'] for r in runs]; out = []
    for h in hn:
        for p in h.parameters(): p.requires_grad_(False)
    for s in range(ns):
        torch.manual_seed(1000 + s); p2 = PsiPar(); e2 = ENet(width, layers)
        with torch.no_grad():
            p2.logA.copy_(torch.tensor(float(np.random.uniform(-1, 1))))
            p2.raw.copy_(torch.tensor(float(np.random.uniform(-1, 2))))
        o = optim.Adam([{'params': p2.parameters(), 'lr': lr * 10},
                        {'params': e2.parameters(), 'lr': lr}])
        for _ in range(ep):
            o.zero_grad()
            L = sum(torch.mean(resid(hn[i], p2, e2, coll(150), W[i]) ** 2) for i in range(len(W)))
            L.backward(); o.step()
        out.append(p2.ab())
    for h in hn:
        for p in h.parameters(): p.requires_grad_(True)
    return out

def train_free(runs, ea, ep, lr, width, layers, seed):
    torch.manual_seed(seed + 7); W = [r['w'] for r in runs]
    td = [r['td'] for r in runs]; hd = [r['hd'] for r in runs]
    hn = [HNet(width, layers) for _ in W]; psi = PsiFree(width, layers); en = ENet(width, layers)
    oA = optim.Adam([p for h in hn for p in h.parameters()], lr=lr)
    for _ in range(ea):
        oA.zero_grad()
        L = sum(torch.mean((hn[i](td[i], 1.0) - hd[i]) ** 2) for i in range(len(W)))
        L.backward(); oA.step()
    params = [p for h in hn for p in h.parameters()] + list(psi.parameters()) + list(en.parameters())
    o = optim.Adam(params, lr=lr)
    for _ in range(ep):
        o.zero_grad(); Ld = Lp = 0.0
        for i in range(len(W)):
            Ld = Ld + torch.mean((hn[i](td[i], 1.0) - hd[i]) ** 2)
            Lp = Lp + torch.mean(resid(hn[i], psi, en, coll(150), W[i]) ** 2)
        (Ld + Lp).backward(); o.step()
    return dict(hn=hn, psi=psi, en=en)

def horizon_w(hn, w0, w1):
    tau = np.linspace(0, 1, 300); tt = torch.tensor(tau, dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        h = [hn[0](tt, 1.0).numpy().ravel(), hn[1](tt, 1.0).numpy().ravel()]
    c = np.abs((w0 ** 2) * h[0] ** 3 - (w1 ** 2) * h[1] ** 3)
    tot = float(np.trapezoid(c, tau)); e = tau < 0.2
    frac = float(np.trapezoid(c[e], tau[e]) / tot) if tot > 0 else float('nan')
    hz = float(tau[np.argmax(c < 0.1)]) if (c < 0.1).any() else 1.0
    return dict(frac_early=frac, horizon=hz, c=c, tau=tau)

# ---- the honest verdict: reads the real h-fit AND the consistency ratio ----
def verdict(ms, hor, fit_errs=None, Ld_A=None, Ld_C=None):
    A = np.array([a for a, _ in ms]); d = np.array([b for _, b in ms])
    arel = np.std(A) / max(abs(np.median(A)), 1e-6)
    drel = np.std(d) / max(abs(np.median(d)), 1e-6)
    v = {}
    worst = max(fit_errs) if fit_errs else None
    if worst is None:
        v['h'] = 'HIGH — fit to data by construction'; h_fit_bad = False
    elif worst < 15:
        v['h'] = 'HIGH — h tracks the data (worst run %.1f%%)' % worst; h_fit_bad = False
    elif worst < 40:
        v['h'] = 'MEDIUM — physics bent h off the data somewhat (worst run %.1f%%)' % worst; h_fit_bad = True
    else:
        v['h'] = 'LOW — physics DRAGGED h off the data (worst run %.1f%%): the ODE cannot make these traces' % worst; h_fit_bad = True
    cons = None
    if Ld_A is not None and Ld_C is not None and Ld_A > 1e-12:
        cons = Ld_C / Ld_A
        if cons > 5:
            v['consistency'] = 'FLAG — data INCONSISTENT with the ODE (joint data-loss %.0fx the data-only fit)' % cons
        elif cons > 2:
            v['consistency'] = 'WARN — physics strains the data fit (joint data-loss %.1fx data-only)' % cons
        else:
            v['consistency'] = 'OK — data consistent with the ODE (joint data-loss %.1fx data-only)' % cons
    inconsistent = (cons is not None and cons > 2) or (worst is not None and worst > 40)
    v['E'] = 'MEDIUM-HIGH — slope of h'
    v['combined'] = 'HIGH — what the physics loss pins' if not inconsistent else \
                    'COMPROMISE — physics & data disagree, so the split is suspect'
    v['Psi amplitude'] = 'MEDIUM (reproducible across restarts)' if arel < 0.3 else 'LOW (multi-start spread large)'
    v['Psi decay'] = 'UNIDENTIFIABLE — a prior-driven extrapolation, NOT a measurement' if drel > 0.5 \
                     else 'LOW-MEDIUM (treat cautiously)'
    note = ("Information horizon: %.0f%% of the 2-run Psi leverage sits in tau<0.2." % (100 * hor['frac_early'])
            if hor else "Single run -> no multi-run lever on Psi.")
    bad = (cons is not None and cons > 5) or (worst is not None and worst > 50)
    warn = inconsistent or (drel > 0.5)
    level = 'bad' if bad else ('warn' if warn else 'ok')
    return v, arel, drel, note, level

def sec(i, t, s=""):
    return ('<div class="sec"><div class="sec-i">%s</div><div class="sec-t">%s</div>'
            '<div class="sec-s">%s</div></div>') % (i, t, s)

def note(html, amber=False):
    return '<div class="note%s">%s</div>' % ("-a" if amber else "", html)

def status_card(level, title, lines):
    body = "".join("<li>%s</li>" % x for x in lines)
    return '<div class="status status-%s"><div class="st-t">%s</div><ul>%s</ul></div>' % (level, title, body)

def fig():
    f, a = plt.subplots(figsize=(6.4, 3.7)); return f, a

# ============================== synthetic sandbox ==============================
def build_demo(psi_A, psi_d, E_B, E_d, rpm_a, rpm_b, n_meas, noise, seed):
    np.random.seed(seed); torch.manual_seed(seed)
    W = [rpm_a / rpm_a, rpm_b / rpm_a]
    Pt = lambda t: psi_A * np.exp(-psi_d * t)
    Et = lambda t: E_B * np.exp(-E_d * t)
    tau = np.linspace(0, 1, 500); runs = []; edges = np.linspace(0, 1, n_meas + 1)
    for w in W:
        s = solve_ivp(lambda t, h, w=w: [-(w ** 2) * Pt(t) * h[0] ** 3 - Et(t)],
                      (0, 1), [1.0], t_eval=tau, method='RK45')
        tg = np.array([np.random.uniform(edges[k], edges[k + 1]) for k in range(n_meas)])
        idx = np.sort(np.unique([np.argmin(np.abs(tau - t)) for t in tg])); idx[-1] = len(tau) - 1
        ht = s.y[0][idx]
        hm = np.clip(ht + noise * ht * np.random.normal(0, 1, len(idx)), 1e-4, None)
        runs.append(dict(td=torch.tensor(tau[idx], dtype=torch.float32).reshape(-1, 1),
                         hd=torch.tensor(hm, dtype=torch.float32).reshape(-1, 1),
                         h_meas=hm, tau_s=tau[idx], w=float(w), h_true=s.y[0]))
    return runs, dict(Psi=Pt(tau), Et=Et(tau))

# ============================== manual / CSV parsing ==============================
EXAMPLE_CSV = ("run_id,t,h,rpm\n0,0,1200,1000\n0,10,430,1000\n0,20,180,1000\n0,30,90,1000\n"
               "1,0,1200,6000\n1,10,250,6000\n1,20,70,6000\n1,30,20,6000")
EXAMPLE_DF = pd.DataFrame([
    {"run_id": 0, "t": 0,  "t": 0,  "h": 1200, "rpm": 1000},
    {"run_id": 0, "t": 10, "h": 430,  "rpm": 1000},
    {"run_id": 0, "t": 20, "h": 180,  "rpm": 1000},
    {"run_id": 0, "t": 30, "h": 90,   "rpm": 1000},
    {"run_id": 1, "t": 0,  "h": 1200, "rpm": 6000},
    {"run_id": 1, "t": 10, "h": 250,  "rpm": 6000},
    {"run_id": 1, "t": 20, "h": 70,   "rpm": 6000},
    {"run_id": 1, "t": 30, "h": 20,   "rpm": 6000},
])

def _rows_from_text(text):
    rows = []
    try:
        rd = csv.DictReader(io.StringIO(text))
        for raw in rd:
            d = {k.strip().strip('﻿'): (v.strip() if isinstance(v, str) else v) for k, v in raw.items()}
            try:
                t = float(d['t']); h = float(d['h'])
            except Exception:
                continue
            rid_s = d.get('run_id', '')
            rid = int(float(rid_s)) if rid_s not in ('', None) else 0
            rpm_s = d.get('rpm', '')
            rpm = float(rpm_s) if rpm_s not in ('', None) else None
            rows.append((rid, t, h, rpm))
    except Exception:
        return []
    return rows

def _rows_from_df(df):
    rows = []; cols = [str(c).strip().lower() for c in df.columns]
    for r in df.itertuples(index=False):
        d = dict(zip(cols, r))
        try:
            t = float(d['t']); h = float(d['h'])
            if math.isnan(t) or math.isnan(h): continue
        except Exception:
            continue
        rid = d.get('run_id', 0)
        try: rid = int(float(rid))
        except Exception: rid = 0
        rpm = d.get('rpm', None)
        try: rpm = float(rpm)
        except Exception: rpm = None
        if rpm is not None and math.isnan(rpm): rpm = None
        rows.append((rid, t, h, rpm))
    return rows

def _build_manual(rows, auto_hw, hw_global, auto_tr, t_ref_manual, default_rpm, rpm_ref):
    groups = {}; order = []
    for rid, t, h, rpm in rows:
        if rid not in groups:
            groups[rid] = dict(t=[], h=[], rpm=rpm); order.append(rid)
        else:
            if rpm is not None: groups[rid]['rpm'] = rpm
        groups[rid]['t'].append(t); groups[rid]['h'].append(h)
    if not groups:
        return None, 'no parseable rows (need columns t and h)'
    t_ref = max(max(g['t']) for g in groups.values()) if auto_tr else float(t_ref_manual)
    if t_ref <= 0:
        return None, 't_ref must be > 0'
    runs = []; preview = []
    for rid in order:
        g = groups[rid]; o = np.argsort(g['t'])
        t_raw = np.array(g['t'])[o]; h_raw = np.array(g['h'])[o]
        if len(t_raw) < 2:
            return None, 'run %d has < 2 points' % rid
        hw = float(h_raw[0]) if auto_hw else float(hw_global)
        if hw <= 0:
            return None, 'run %d h_wet must be > 0' % rid
        tau = t_raw / t_ref; h_tilde = h_raw / hw
        rpm = g['rpm'] if g['rpm'] is not None else float(default_rpm)
        runs.append(dict(td=torch.tensor(tau, dtype=torch.float32).reshape(-1, 1),
                         hd=torch.tensor(h_tilde, dtype=torch.float32).reshape(-1, 1),
                         h_meas=h_tilde, tau_s=tau, w=float(rpm / rpm_ref), rpm=rpm))
        preview.append(dict(run=rid, n=len(t_raw), rpm=rpm, w=round(rpm / rpm_ref, 3),
                            h_wet=round(hw, 4), tau_min=round(float(tau[0]), 3),
                            tau_max=round(float(tau[-1]), 3)))
    return runs, preview

# ============================== sidebar ==============================
st.sidebar.markdown("### ⚙️ Controls")
source = st.sidebar.radio("Data source", ["Synthetic", "Manual / CSV"], index=0,
    help="Synthetic = app generates the hidden truth. Manual / CSV = your own thickness-vs-time.")
SRC_SYN = (source == "Synthetic")

with st.sidebar.expander("🧪 Physics", expanded=True):
    psi_A = st.slider("Psi_A · convective strength", 0.1, 3.0, 1.2, 0.05)
    psi_d = st.slider("Psi decay", 0.5, 6.0, 3.0, 0.1)
    E_B   = st.slider("E_B · evaporation strength", 0.5, 6.0, 3.0, 0.1)
    E_d   = st.slider("E decay", 0.5, 6.0, 3.5, 0.1)
    rpm_a = st.slider("Run A · RPM", 1000, 6000, 3000, 100)
    rpm_b = st.slider("Run B · RPM", 1000, 6000, 4500, 100)
with st.sidebar.expander("📡 Synthetic data", expanded=SRC_SYN):
    n_meas = st.slider("Measurements / run", 4, 24, 8)
    noise  = st.slider("Noise sigma", 0.0, 0.10, 0.02, 0.005)
    seed   = st.number_input("Seed", 0, 999, 42)
with st.sidebar.expander("🧠 Training", expanded=True):
    ea = st.slider("Phase A epochs", 100, 1500, 400, 50)
    eb = st.slider("Phase B epochs", 100, 1500, 500, 50)
    ec = st.slider("Phase C epochs", 100, 1500, 400, 50)
    ef = st.slider("Free-Psi epochs", 200, 2000, 900, 100)
    lr = st.select_slider("Learning rate", [5e-4, 1e-3, 2e-3, 5e-3], value=1e-3)
    width  = st.slider("Hidden width", 16, 64, 32, 8)
    layers = st.slider("Hidden layers", 2, 5, 3)
    wd = st.slider("W_data", 0.1, 5.0, 1.0, 0.1)
    wp = st.slider("W_physics", 0.1, 5.0, 1.0, 0.1)
    ns = st.slider("Multi-start restarts", 2, 10, 4)

# ============================== tabs ==============================
tb = st.tabs(["🧪 Physics", "📥 Manual / CSV", "📡 Data", "🧠 Train", "📊 Results", "ℹ️ Model"])

# ---- 0: Physics (forward sim from sliders) ----
with tb[0]:
    st.markdown(sec("01", "Forward simulator", "Integrate the ground-truth ODE at the two spin speeds (synthetic physics)."))
    tau = np.linspace(0, 1, 500); w_ref = rpm_a
    c1, c2 = st.columns(2)
    with c1:
        f, a = fig()
        for rpm, col in ((rpm_a, '#22d3ee'), (rpm_b, '#fbbf24')):
            w = rpm / w_ref
            s = solve_ivp(lambda t, h, w=w: [-(w ** 2) * psi_A * np.exp(-psi_d * t) * h[0] ** 3 - E_B * np.exp(-E_d * t)],
                          (0, 1), [1.0], t_eval=tau, method='RK45')
            a.plot(tau, s.y[0], color=col, lw=2.4, label='%d RPM' % rpm)
        a.set_xlabel('τ'); a.set_ylabel('ĥ'); a.set_title('Thinning ĥ(τ)'); a.legend(frameon=False); a.grid(alpha=.3); st.pyplot(f)
    with c2:
        f, a = fig()
        a.plot(tau, psi_A * np.exp(-psi_d * tau), color='#22d3ee', lw=2.4, label='Ψ(τ)')
        a.plot(tau, E_B * np.exp(-E_d * tau), color='#fbbf24', lw=2.4, label='Ẽ(τ)')
        a.set_xlabel('τ'); a.set_title('Latent Ψ & Ẽ'); a.legend(frameon=False); a.grid(alpha=.3); st.pyplot(f)
    st.caption("K̃(τ) = (ω/ω_ref)²·Ψ(τ). Higher spin -> stronger convective thinning.")

# ---- 1: Manual / CSV (sets session_state['manual']) ----
with tb[1]:
    st.markdown(sec("02", "Your thickness data",
                    "Enter h vs t per run. Normalized internally: h̃ = h/h_wet, τ = t/t_ref. "
                    "Give at least 2 runs at different RPM for the Ψ/ split to be identifiable in principle."))
    upl = st.file_uploader("Upload a .csv  (columns: run_id, t, h, rpm)", type=["csv"], key="manual_csv")
    txt = st.text_area("or paste CSV text", value="", key="manual_csv_text",
                       height=130, placeholder=EXAMPLE_CSV)
    st.markdown("**or edit the table**")
    df = st.data_editor(EXAMPLE_DF, key="manual_df", num_rows="dynamic",
                        use_container_width=True, hide_index=True)
    cc1, cc2 = st.columns(2)
    with cc1:
        auto_hw = st.checkbox("h_wet = each run's first thickness", value=True)
        if not auto_hw:
            hw_global = st.number_input("global h_wet", 1e-6, 1e9, 1.0, format="%.4f")
        else:
            hw_global = 1.0
    with cc2:
        auto_tr = st.checkbox("t_ref = max time across runs", value=True)
        if not auto_tr:
            t_ref_manual = st.number_input("t_ref", 1e-6, 1e9, 1.0, format="%.4f")
        else:
            t_ref_manual = 1.0
    default_rpm = st.number_input("default RPM (if a run has no rpm column)", 100, 10000, 3000, 100)
    if st.button("⬇  Load / refresh from the inputs above", use_container_width=True, type="primary"):
        if upl is not None:
            rows = _rows_from_text(upl.getvalue().decode("utf-8-sig"))
        elif txt.strip():
            rows = _rows_from_text(txt)
        else:
            rows = _rows_from_df(df)
        built, info = _build_manual(rows, auto_hw, hw_global, auto_tr, t_ref_manual, default_rpm, rpm_a)
        if built is None:
            st.error("Could not load: %s" % info)
        else:
            st.session_state['manual'] = dict(runs=built, preview=info)
            st.success("Loaded %d run(s). Now hit **Train**." % len(built))
    if st.session_state.get('manual') and st.session_state['manual'].get('preview'):
        st.markdown("**Loaded (normalized):**")
        st.dataframe(pd.DataFrame(st.session_state['manual']['preview']),
                     use_container_width=True, hide_index=True)

# ---- resolve runs (synthetic or the manual data just loaded) ----
if SRC_SYN:
    runs, meta = build_demo(psi_A, psi_d, E_B, E_d, rpm_a, rpm_b, n_meas, noise, seed)
    has_truth = True
else:
    m = st.session_state.get('manual')
    if m is None or not m.get('runs'):
        st.warning("Manual mode: open the **Manual / CSV** tab, enter data, click **Load**, then re-run.")
        st.stop()
    runs = m['runs']; meta = {}; has_truth = False

# ---- 2: Data ----
with tb[2]:
    st.markdown(sec("03", "What the PINN sees",
                    "Sparse (normalized) thickness per run + dense unlabeled collocation points."))
    cols = st.columns(min(len(runs), 2))
    for i in range(min(len(runs), 2)):
        with cols[i]:
            f, a = fig(); col = ['#22d3ee', '#fbbf24'][i % 2]
            if has_truth:
                a.plot(runs[i].get('tau_s', np.linspace(0, 1, 500)) if False else np.linspace(0, 1, 500),
                       runs[i]['h_true'], color=col, lw=2.2, alpha=.5, label='true ĥ')
            a.scatter(runs[i]['tau_s'], runs[i]['h_meas'], color=col, s=46, zorder=5, label='data')
            a.set_xlabel('τ'); a.set_ylabel('ĥ'); a.set_title('Run %d · %s RPM' % (i, runs[i].get('rpm', '?')))
            a.legend(frameon=False, fontsize=8); a.grid(alpha=.3); st.pyplot(f)

# ---- 3: Train ----
with tb[3]:
    st.markdown(sec("04", "Train the inverse model",
                    "Constrained Ψ = A·exp(−d·τ) + shared Ẽ. Records the data-loss before and after "
                    "the physics is enforced (for the consistency flag)."))
    if st.button("🧠  Train inverse model", use_container_width=True, type="primary"):
        with st.spinner("Training constrained model, multi-start, and free-Ψ diagnostic…"):
            param = train_parametric(runs, ea, eb, ec, lr, wd, wp, width, layers, seed)
            ms = multistart(param['hn'], runs, ns, 250, lr, width, layers)
            free = train_free(runs, 300, ef, lr, width, layers, seed)
        st.session_state['train_res'] = dict(param=param, ms=ms, free=free)
        st.success("Training complete — see **Results**.")
    if st.session_state.get('train_res'):
        h = st.session_state['train_res']['param']
        f, a = fig(); a.plot(h.get('hist', [])); a.set_title("(train stored)"); st.pyplot(f) if False else None

# ---- 4: Results ----
with tb[4]:
    st.markdown(sec("05", "Inverse recovery",
                    "Headline Ψ/Ẽ/h, the identifiable combined term, the information horizon, "
                    "and the honest trust verdict."))
    tr = st.session_state.get('train_res')
    if tr is None:
        st.info("Train first.")
    else:
        param, ms, free = tr['param'], tr['ms'], tr['free']
        tau_d = np.linspace(0, 1, 300); tt = torch.tensor(tau_d, dtype=torch.float32).reshape(-1, 1)
        with torch.no_grad():
            Pp = param['psi'](tt).numpy().ravel(); Ep = param['en'](tt).numpy().ravel()
            hp = [param['hn'][i](tt, 1.0).numpy().ravel() for i in range(len(runs))]
        m1, m2, m3, m4 = st.columns(4)
        if has_truth:
            m1.metric("Ψ(τ) error", "%.1f%%" % rel(Pp, meta['Psi']))
            m2.metric("Ẽ(τ) error", "%.1f%%" % rel(Ep, meta['Et']))
        else:
            m1.metric("Ψ(τ)", "no truth"); m2.metric("Ẽ(τ)", "no truth")
        h0e = rel(param['hn'][0](runs[0]['td'], 1.0).detach().numpy().ravel(), runs[0]['h_meas'])
        m3.metric("ĥ run A (fit)", "%.1f%%" % h0e)
        if len(runs) > 1:
            h1e = rel(param['hn'][1](runs[1]['td'], 1.0).detach().numpy().ravel(), runs[1]['h_meas'])
            m4.metric("ĥ run B (fit)", "%.1f%%" % h1e)
        else:
            h1e = None; m4.metric("ĥ run B", "—")
        c1, c2 = st.columns(2)
        with c1:
            f, a = fig()
            if has_truth:
                a.plot(tau_d, meta['Psi'], color='#22d3ee', lw=2.4, label='true Ψ')
                a.plot(tau_d, meta['Et'], color='#fbbf24', lw=2.4, label='true Ẽ')
            a.plot(tau_d, Pp, '--', color='#22d3ee', lw=2, label='pred Ψ')
            a.plot(tau_d, Ep, '--', color='#fbbf24', lw=2, label='pred Ẽ')
            a.set_title('Shared Ψ & Ẽ'); a.legend(frameon=False, fontsize=8); a.grid(alpha=.3); st.pyplot(f)
        with c2:
            f, a = fig()
            for i in range(len(runs)):
                cp = (runs[i]['w'] ** 2) * Pp * hp[i] ** 3 + Ep
                a.plot(tau_d, cp, '--', color=['#22d3ee', '#fbbf24'][i % 2], lw=2, label='pred run%d' % i)
            a.set_title('Combined ODE term (identifiable part)'); a.legend(frameon=False, fontsize=8); a.grid(alpha=.3); st.pyplot(f)
        hor = horizon_w(param['hn'], runs[0]['w'], runs[1]['w']) if len(runs) > 1 else None
        if hor:
            f, a = fig()
            a.plot(hor['tau'], hor['c'], color='#cbd5e1', lw=2); a.axvline(0.2, color='#ff6b6b', ls=':')
            a.set_title('2-run Ψ leverage c(τ): %.0f%% in τ<0.2' % (100 * hor['frac_early'])); a.grid(alpha=.3); st.pyplot(f)
        with torch.no_grad():
            Pf = free['psi'](tt).numpy().ravel()
        f, a = fig()
        if has_truth:
            a.plot(tau_d, meta['Psi'], color='#22d3ee', lw=2.4, label='true Ψ')
        a.plot(tau_d, Pf, '--', color='#ff6b6b', lw=2, label='free Ψ (diagnostic)')
        a.set_title('Free-Ψ diagnostic — fits the sum, hallucinates the split'); a.legend(frameon=False, fontsize=8); a.grid(alpha=.3); st.pyplot(f)
        # ---- honest verdict (reads the real h-fit + the consistency ratio) ----
        fit_errs = [h0e] + ([h1e] if h1e is not None else [])
        Ld_A = param.get('Ld_A'); Ld_C = param.get('Ld_C')
        v, arel, drel, note_txt, level = verdict(ms, hor, fit_errs=fit_errs, Ld_A=Ld_A, Ld_C=Ld_C)
        flag = {'ok': '✅ ', 'warn': '⚠️ ', 'bad': '🚨 '}[level]
        st.markdown(status_card(level, flag + "Trust verdict",
                                ["%s: %s" % (k, val) for k, val in v.items()] + [note_txt]))

# ---- 5: Model ----
with tb[5]:
    st.markdown(sec("06", "Model & honest limits", ""))
    st.markdown(note(
        "<b>What is recoverable from sparse thickness-vs-time:</b> thickness (a few %), "
        "evaporation Ẽ (the slope), and the <b>combined</b> term w²Ψĥ³+. With a constrained Ψ you also "
        "get the right decaying <i>shape</i> and right-order amplitude."))
    st.markdown(note(
        "<b>What is NOT recoverable:</b> the viscosity <i>decay rate</i>. Three independent diagnostics "
        "agree — multi-start spread, the free-Ψ mirror image, and the information horizon (≈95% of the "
        "2-run Ψ leverage sits in τ&lt;0.2). The new <b>consistency flag</b> additionally tells you when "
        "your traces are inconsistent with the ODE at all (🚨): that means no amount of tuning will "
        "recover a physical split, because the ODE cannot produce those traces.", amber=True))
    st.markdown(note(
        "<b>How to read the verdict:</b> ✅ data consistent & decay still unidentifiable (normal); "
        "⚠️ decay unidentifiable or mild physics/data strain; 🚨 the joint physics loss had to drag h off "
        "the data (joint data-loss ≫ data-only fit) → the split is a prior-driven extrapolation, not a "
        "measurement. To recover viscosity dynamics, sample τ&lt;0.2 densely and/or add a concentration/"
        "rheology observable.", amber=True))
