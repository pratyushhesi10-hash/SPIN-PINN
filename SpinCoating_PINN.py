# app.py  —  Thin-Film Inverse Lab  (per-run normalization + normalization gate)
# Run:  streamlit run app.py     Deps: streamlit numpy scipy torch matplotlib pandas
import io, csv, math
import numpy as np
import pandas as pd
import streamlit as st
import torch, torch.nn as nn, torch.optim as optim
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="Thin-Film Inverse Lab", page_icon="◈", layout="wide")

# ── crafted instrument theme (ambient layers + display/body/mono pairing) ──────
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=IBM+Plex+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
:root{--ink:#e9edf2;--bg:#0c1117;--panel:rgba(255,255,255,.035);--line:rgba(233,237,242,.10);
--c:#34d6cf;--a:#f0a93b;--r:#ff6b6b;--mute:#8a97a6;
--disp:'Fraunces',serif;--body:'IBM Plex Sans',sans-serif;--mono:'JetBrains Mono',monospace;}
html,body,[class*="css"]{font-family:var(--body);}
[data-testid="stAppViewContainer"]{background:
  radial-gradient(900px 520px at 92% -6%, rgba(52,214,207,.10), transparent 60%),
  radial-gradient(820px 520px at -6% 106%, rgba(240,169,59,.07), transparent 60%),
  var(--bg)!important;}
[data-testid="stAppViewContainer"]::before{content:"";position:fixed;inset:0;z-index:0;pointer-events:none;
  background-image:radial-gradient(rgba(233,237,242,.05) 1px,transparent 1px);background-size:26px 26px;}
.grain{position:fixed;inset:0;z-index:0;pointer-events:none;opacity:.04;mix-blend-mode:overlay;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");}
.scan{position:fixed;left:0;right:0;height:1px;z-index:0;pointer-events:none;opacity:.35;
  background:linear-gradient(90deg,transparent,var(--c),transparent);animation:scan 9s linear infinite;}
@keyframes scan{0%{top:-2%}100%{top:102%}}
@keyframes rise{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.3;transform:scale(.7)}}
[data-testid="stMain"],[data-testid="stMainBlockContainer"]{background:transparent;}
[data-testid="stSidebar"]{background:rgba(10,14,20,.82)!important;border-right:1px solid var(--line);}
[data-testid="stSidebar"] *{color:var(--ink)!important;}
[data-testid="stHeader"]{background:var(--bg)!important;}
[data-testid="stToolbar"]{filter:invert(1) grayscale(1) brightness(1.7);}
#MainMenu{visibility:hidden;} footer{visibility:hidden;}
.block-container{padding-top:2rem;padding-bottom:2rem;max-width:1200px;}
.stButton>button{background:var(--panel)!important;border:1px solid var(--line)!important;color:var(--ink)!important;}
div[data-testid="stSlider"] [role="slider"]{background-color:var(--c)!important;border-color:var(--c)!important;}
div[data-testid="stSlider"] *{color:var(--ink)!important;}
div[data-baseweb="tab-highlight"]{background-color:var(--c)!important;}
.kicker{font-family:var(--mono);font-size:.7rem;letter-spacing:.26em;text-transform:uppercase;color:var(--c);margin:0 0 18px;}
.title{font-family:var(--disp);font-weight:600;font-optical-sizing:auto;line-height:.94;letter-spacing:-.02em;
  font-size:clamp(2.6rem,6.2vw,5.2rem);margin:0;color:var(--ink);}
.title em{font-style:italic;color:var(--a);}
.lede{max-width:64ch;color:var(--mute);font-weight:300;font-size:1.05rem;margin:18px 0 0;line-height:1.6;}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:18px;}
.chip{font-family:var(--mono);font-size:.7rem;padding:6px 12px;border-radius:999px;border:1px solid var(--line);
  color:var(--mute);background:var(--panel);transition:.2s;}
.chip:hover{border-color:var(--c);color:var(--c);transform:translateY(-2px);}
.chip-c{border-color:rgba(52,214,207,.4);color:var(--c);} .chip-a{border-color:rgba(240,169,59,.4);color:var(--a);}
.strip{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:26px 0 4px;padding:12px 16px;
  border:1px solid var(--line);border-radius:12px;background:var(--panel);font-family:var(--mono);font-size:.72rem;}
.strip .k{color:var(--mute);letter-spacing:.12em;text-transform:uppercase;}
.strip .v{color:var(--ink);} .strip .sep{color:var(--line);}
.g-ok{color:var(--c);} .g-warn{color:var(--a);} .g-block{color:var(--r);}
.sec{margin:6px 0 16px;animation:rise .55s both;}
.sec-i{font-family:var(--mono);font-size:.68rem;letter-spacing:.2em;color:var(--a);text-transform:uppercase;}
.sec-t{font-family:var(--disp);font-weight:600;font-size:clamp(1.4rem,3vw,2rem);color:var(--ink);line-height:1.05;margin:2px 0;}
.sec-s{color:var(--mute);font-weight:300;max-width:70ch;}
.blk{animation:rise .55s both;}
.status{display:flex;gap:14px;align-items:flex-start;padding:14px 18px;border-radius:12px;border:1px solid var(--line);
  border-left:4px solid var(--mute);background:var(--panel);margin:6px 0;}
.status.ok{border-left-color:var(--c);} .status.warn{border-left-color:var(--a);} .status.block{border-left-color:var(--r);}
.status .dot{width:9px;height:9px;border-radius:50%;margin-top:6px;flex:none;background:var(--mute);animation:pulse 1.6s infinite;}
.status.ok .dot{background:var(--c);} .status.warn .dot{background:var(--a);} .status.block .dot{background:var(--r);}
.status .st-t{font-family:var(--mono);font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;margin-bottom:6px;}
.status.ok .st-t{color:var(--c);} .status.warn .st-t{color:var(--a);} .status.block .st-t{color:var(--r);}
.status ul{margin:0;padding-left:16px;color:var(--ink);font-size:.86rem;line-height:1.5;}
.status li{margin:2px 0;}
.note{border-left:3px solid var(--c);background:rgba(52,214,207,.06);border-radius:0 10px 10px 0;padding:12px 16px;
  color:var(--ink);font-size:.9rem;} .note b{color:var(--c);}
[data-testid="stMetric"]{background:linear-gradient(160deg,rgba(255,255,255,.05),rgba(255,255,255,.012));
  border:1px solid var(--line);border-left:3px solid var(--c);border-radius:14px;padding:14px 16px;transition:.22s;}
[data-testid="stMetric"]:hover{transform:translateY(-3px);border-left-color:var(--a);box-shadow:0 16px 34px -22px #000;}
[data-testid="stMetricValue"]{font-family:var(--disp)!important;font-weight:600;}
[data-testid="stMetricLabel"]{font-family:var(--mono)!important;text-transform:uppercase;letter-spacing:.1em;font-size:.62rem!important;color:var(--mute)!important;}
.stTabs [data-baseweb="tab"]{font-family:var(--body)!important;font-weight:600!important;}
.stTabs [data-baseweb="tab"]:hover{color:var(--c)!important;}
.stTabs [data-baseweb="tab-list"]{border-bottom:1px solid var(--line)!important;}
.stButton>button{border-radius:12px!important;font-weight:600!important;transition:.18s!important;}
.stButton>button:hover{transform:translateY(-2px);box-shadow:0 12px 26px -16px var(--c);}
hr{border-color:var(--line)!important;}
</style>
<div class="grain"></div><div class="scan"></div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="kicker">physics-informed inverse lab · thin-film drying</div>
<h1 class="title">Read the film.<br><em>Infer</em> the physics.</h1>
<p class="lede">A PINN recovers the hidden viscosity Ψ(τ) and evaporation Ẽ(τ) from sparse thickness
data. Paste <b>raw dimensional</b> measurements — the lab auto‑normalizes each run with its own wet
thickness, then a <b>normalization gate</b> checks h̃(0)≈1 and τ∈[0,1] and refuses to train with a
plain‑English diagnosis if anything is off.</p>
<div class="chips">
  <span class="chip chip-c">dĥ/dτ = −w²Ψĥ³ − Ẽ</span>
  <span class="chip">per-run h_wet · auto-detected</span>
  <span class="chip chip-a">normalization gate</span>
  <span class="chip">trust verdict</span>
</div>
""", unsafe_allow_html=True)

# ── helpers ───────────────────────────────────────────────────────────────────
def rel(p, t):
    p = np.asarray(p, float).ravel(); t = np.asarray(t, float).ravel()
    return float(np.mean(np.abs(p - t) / (np.abs(t) + 1e-8)) * 100)

def _group(tuples):
    groups, order = {}, []
    for rid, t, h, rpm in tuples:
        if rid not in groups:
            groups[rid] = dict(id=rid, t=[], h=[], rpm=rpm); order.append(rid)
        elif rpm is not None:
            groups[rid]['rpm'] = rpm
        groups[rid]['t'].append(t); groups[rid]['h'].append(h)
    out = []
    for rid in order:
        g = groups[rid]; idx = np.argsort(g['t'])
        out.append(dict(id=rid, t=np.array(g['t'])[idx], h=np.array(g['h'])[idx], rpm=g['rpm']))
    return out

def parse_text(text):
    tuples = []
    try:
        rd = csv.DictReader(io.StringIO(text))
        for raw in rd:
            d = {k.strip().lstrip("﻿"): (v.strip() if isinstance(v, str) else v) for k, v in raw.items()}
            try: t = float(d['t']); h = float(d['h'])
            except Exception: continue
            rid = int(float(d.get('run_id', 0) or 0))
            rv = d.get('rpm', ''); rpm = float(rv) if rv not in ('', None) else None
            tuples.append((rid, t, h, rpm))
    except Exception:
        return []
    return _group(tuples)

def parse_df(df):
    tuples = []
    for _, row in df.iterrows():
        try: t = float(row['t']); h = float(row['h'])
        except Exception: continue
        if math.isnan(t) or math.isnan(h): continue
        rid = int(row['run_id']) if 'run_id' in df.columns and not (isinstance(row['run_id'], float) and math.isnan(row['run_id'])) else 0
        rpm = row['rpm'] if 'rpm' in df.columns else None
        if isinstance(rpm, float) and math.isnan(rpm): rpm = None
        tuples.append((rid, t, h, rpm))
    return _group(tuples)

# pre-loaded example = the exact 1000/6000 case that broke global normalization
EXAMPLE_ROWS = [
    {'run_id':0,'t':0,'h':1200,'rpm':1000},{'run_id':0,'t':2,'h':792,'rpm':1000},
    {'run_id':0,'t':4,'h':540,'rpm':1000},{'run_id':0,'t':7,'h':360,'rpm':1000},
    {'run_id':0,'t':12,'h':216,'rpm':1000},{'run_id':0,'t':20,'h':132,'rpm':1000},
    {'run_id':0,'t':30,'h':90,'rpm':1000},{'run_id':0,'t':40,'h':72,'rpm':1000},
    {'run_id':1,'t':0,'h':850,'rpm':6000},{'run_id':1,'t':2,'h':425,'rpm':6000},
    {'run_id':1,'t':4,'h':255,'rpm':6000},{'run_id':1,'t':7,'h':145,'rpm':6000},
    {'run_id':1,'t':12,'h':77,'rpm':6000},{'run_id':1,'t':20,'h':43,'rpm':6000},
    {'run_id':1,'t':30,'h':26,'rpm':6000},{'run_id':1,'t':40,'h':13,'rpm':6000}]
EXAMPLE_DF = pd.DataFrame(EXAMPLE_ROWS)
EXAMPLE = EXAMPLE_DF.to_csv(index=False)
DEFAULT_RAW = _group([(r['run_id'], r['t'], r['h'], r['rpm']) for r in EXAMPLE_ROWS])

def build_runs(raw_runs, auto_h, hw_manual, auto_t, tr):
    rpm_vals = [r['rpm'] for r in raw_runs if r['rpm'] is not None]
    rpm_ref = min(rpm_vals) if rpm_vals else 1.0
    gmax = max(float(r['t'].max()) for r in raw_runs)
    tr_use = gmax if auto_t else (tr if tr and tr > 0 else gmax)
    runs = []
    for r in raw_runs:
        t, h = r['t'], r['h']; i0 = int(np.argmin(t))
        raw_h0 = float(h[i0]); raw_tmax = float(t.max())
        hw = raw_h0 if auto_h else float(hw_manual.get(r['id'], raw_h0))
        if hw <= 0: hw = raw_h0 if raw_h0 > 0 else 1.0
        w = (r['rpm'] / rpm_ref) if r['rpm'] is not None else 1.0
        tau_s = t / tr_use; h_meas = h / hw
        runs.append(dict(td=torch.tensor(tau_s, dtype=torch.float32).reshape(-1, 1),
                         hd=torch.tensor(h_meas, dtype=torch.float32).reshape(-1, 1),
                         tau_s=tau_s, h_meas=h_meas, w=float(w), rpm=r['rpm'],
                         hw_used=float(hw), tr_used=float(tr_use), raw_h0=raw_h0,
                         raw_tmax=raw_tmax, n_pts=len(tau_s), id=r['id']))
    return runs, float(rpm_ref)

def gate(runs):
    issues, warns, diag = [], [], []
    for r in runs:
        ts, hm, n = r['tau_s'], r['h_meas'], len(r['tau_s'])
        if n < 2:
            issues.append(f"Run {r['id']}: needs ≥2 points (has {n}).")
            diag.append([r['id'], r['rpm'], round(r['hw_used'],1), round(r['tr_used'],1), '—', '—', '—', n]); continue
        i0 = int(np.argmin(ts)); tmin, tmax, h0 = float(ts[i0]), float(ts.max()), float(hm[i0])
        if tmin < -1e-6 or tmax > 1 + 1e-3:
            issues.append(f"Run {r['id']}: τ∈[{tmin:.2f}, {tmax:.2f}] outside [0,1]. Set t_ref ≥ max time (≈{r['raw_tmax']:.0f}) or enable auto t_ref.")
        if abs(h0 - 1.0) > 0.05:
            issues.append(f"Run {r['id']}: h̃(earliest)={h0:.2f} ≠ 1. Set h_wet≈{r['raw_h0']:.0f} for this run, or enable auto h_wet.")
        if (hm <= 0).any(): issues.append(f"Run {r['id']}: h̃≤0 somewhere (check h_wet).")
        if (hm > 1.08).any(): warns.append(f"Run {r['id']}: some h̃>1.08 (h_wet should be the initial/max thickness).")
        diag.append([r['id'], r['rpm'], round(r['hw_used'],1), round(r['tr_used'],1), round(tmin,3), round(tmax,3), round(h0,3), n])
    ws = sorted({round(r['w'], 3) for r in runs})
    if len(ws) < 2:
        warns.append("All runs share one ω scaling → no multi-run lever; the Ψ/Ẽ split is structurally unidentifiable (expected teaching case, not an error).")
    return issues, warns, diag

def sec(i, t, s=""):
    return f'<div class="sec"><div class="sec-i">{i}</div><div class="sec-t">{t}</div><div class="sec-s">{s}</div></div>'

def status_card(level, title, lines):
    body = "".join(f"<li>{x}</li>" for x in lines) if lines else "<li>Normalization looks good — h̃(0)≈1 and τ∈[0,1] for every run.</li>"
    return f'<div class="status {level}"><span class="dot"></span><div><div class="st-t">{title}</div><ul>{body}</ul></div></div>'

# ── networks + training (constrained Ψ + free-Ψ diagnostic + multi-start) ─────
def mlp(h, L):
    lay = [nn.Linear(1, h), nn.Tanh()]
    for _ in range(L - 1): lay += [nn.Linear(h, h), nn.Tanh()]
    return nn.Sequential(*lay, nn.Linear(h, 1))

class HNet(nn.Module):
    def __init__(s, h=32, L=3): super().__init__(); s.net = mlp(h, L); s.sp = nn.Softplus()
    def forward(s, t, h0=1.0): return h0 - t * s.sp(s.net(t))
class PsiPar(nn.Module):                       # constrained A*exp(-d*tau), d>=0
    def __init__(s):
        super().__init__()
        s.logA = nn.Parameter(torch.tensor(0.0))
        s.raw = nn.Parameter(torch.tensor(0.5))
        s.sp = nn.Softplus()
    def forward(s, t): return torch.exp(s.logA - s.sp(s.raw) * t)
    def ab(s): return float(torch.exp(s.logA).item()), float(s.sp(s.raw).item())
class ENet(nn.Module):
    def __init__(s, h=32, L=3): super().__init__(); s.net = mlp(h, L); s.sp = nn.Softplus()
    def forward(s, t): return s.sp(s.net(t))
class PsiFree(nn.Module):
    def __init__(s, h=32, L=3): super().__init__(); s.net = mlp(h, L)
    def forward(s, t): return torch.exp(s.net(t))

def resid(hn, psi, en, t, w):
    t = t.reshape(-1, 1); h = hn(t, 1.0)
    dh = torch.autograd.grad(h, t, grad_outputs=torch.ones_like(h), create_graph=True, retain_graph=True)[0]
    return dh + (w ** 2) * psi(t) * h ** 3 + en(t)
def coll(n):
    t = torch.tensor(np.sort(np.random.uniform(0, 1, n)), dtype=torch.float32).reshape(-1, 1)
    t.requires_grad_(True); return t

def train_parametric(runs, ea, eb, ec, lr, wd, wp, width, layers, seed):
    torch.manual_seed(seed); W = [r['w'] for r in runs]
    td = [r['td'] for r in runs]; hd = [r['hd'] for r in runs]
    hn = [HNet(width, layers) for _ in W]; psi = PsiPar(); en = ENet(width, layers)
    oA = optim.Adam([p for h in hn for p in h.parameters()], lr=lr)
    for _ in range(ea):
        oA.zero_grad(); Ld = sum(torch.mean((hn[i](td[i], 1.0) - hd[i]) ** 2) for i in range(len(W)))
        Ld.backward(); oA.step()
    for h in hn:
        for p in h.parameters(): p.requires_grad_(False)
    oB = optim.Adam([{'params': psi.parameters(), 'lr': lr * 10}, {'params': en.parameters(), 'lr': lr}])
    for _ in range(eb):
        oB.zero_grad(); Lp = sum(torch.mean(resid(hn[i], psi, en, coll(150), W[i]) ** 2) for i in range(len(W)))
        Lp.backward(); oB.step()
    for h in hn:
        for p in h.parameters(): p.requires_grad_(True)
    oC = optim.Adam([{'params': [p for h in hn for p in h.parameters()], 'lr': lr * 0.1},
                     {'params': psi.parameters(), 'lr': lr}, {'params': en.parameters(), 'lr': lr * 0.1}])
    for _ in range(ec):
        oC.zero_grad(); Ld = Lp = 0.0
        for i in range(len(W)):
            Ld = Ld + torch.mean((hn[i](td[i], 1.0) - hd[i]) ** 2)
            Lp = Lp + torch.mean(resid(hn[i], psi, en, coll(150), W[i]) ** 2)
        (wd * Ld + wp * Lp).backward(); oC.step()
    return dict(hn=hn, psi=psi, en=en)

def multistart(hn, runs, ns, ep, lr):
    W = [r['w'] for r in runs]; out = []
    for h in hn:
        for p in h.parameters(): p.requires_grad_(False)
    for s in range(ns):
        torch.manual_seed(1000 + s); p2 = PsiPar(); e2 = ENet(32, 3)
        with torch.no_grad():
            p2.logA.copy_(torch.tensor(float(np.random.uniform(-1, 1))))
            p2.raw.copy_(torch.tensor(float(np.random.uniform(-1, 2))))
        o = optim.Adam([{'params': p2.parameters(), 'lr': lr * 10}, {'params': e2.parameters(), 'lr': lr}])
        for _ in range(ep):
            o.zero_grad(); Lp = sum(torch.mean(resid(hn[i], p2, e2, coll(150), W[i]) ** 2) for i in range(len(W)))
            Lp.backward(); o.step()
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
        oA.zero_grad(); Ld = sum(torch.mean((hn[i](td[i], 1.0) - hd[i]) ** 2) for i in range(len(W)))
        Ld.backward(); oA.step()
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
    with torch.no_grad(): h = [hn[0](tt, 1.0).numpy().ravel(), hn[1](tt, 1.0).numpy().ravel()]
    c = np.abs((w0 ** 2) * h[0] ** 3 - (w1 ** 2) * h[1] ** 3)
    tot = np.trapezoid(c, tau); e = tau < 0.2
    frac = float(np.trapezoid(c[e], tau[e]) / tot) if tot > 0 else float('nan')
    hz = float(tau[np.argmax(c < 0.1)]) if (c < 0.1).any() else 1.0
    return dict(frac_early=frac, horizon=hz, c=c, tau=tau)

def verdict(ms, hor):
    A = np.array([a for a, _ in ms]); d = np.array([b for _, b in ms])
    arel = np.std(A) / max(abs(np.median(A)), 1e-6); drel = np.std(d) / max(abs(np.median(d)), 1e-6)
    v = {'h': 'HIGH — fit to data by construction', 'Ẽ': 'MEDIUM-HIGH — slope of h',
         'combined': 'HIGH — what the physics loss pins',
         'Ψ amplitude': 'MEDIUM (reproducible across restarts)' if arel < 0.3 else 'LOW (multi-start spread large)',
         'Ψ decay': 'UNIDENTIFIABLE — a prior-driven extrapolation, not a measurement' if drel > 0.5 else 'LOW-MEDIUM (treat cautiously)'}
    note = (f"Information horizon: {100*hor['frac_early']:.0f}% of the 2-run Ψ leverage sits in τ<0.2."
            if hor else "Single run → no multi-run lever on Ψ.")
    return v, arel, drel, note

def demo_build(PA, PD, EB, ED, rpm_a, rpm_b, n_meas, noise, seed):
    np.random.seed(seed); torch.manual_seed(seed)
    W = [rpm_a / rpm_a, rpm_b / rpm_a]; rpms = [rpm_a, rpm_b]
    Pt = lambda t: PA * np.exp(-PD * t); Et = lambda t: EB * np.exp(-ED * t)
    te = np.linspace(0, 1, 500); runs = []; edges = np.linspace(0, 1, n_meas + 1)
    for i, w in enumerate(W):
        s = solve_ivp(lambda t, h, w=w: [-(w ** 2) * Pt(t) * h[0] ** 3 - Et(t)], (0, 1), [1.0], t_eval=te, method='RK45')
        tg = np.array([np.random.uniform(edges[k], edges[k + 1]) for k in range(n_meas)])
        idx = np.sort(np.unique([np.argmin(np.abs(te - t)) for t in tg])); idx[-1] = len(te) - 1
        ht = s.y[0][idx]; hm = np.clip(ht + noise * ht * np.random.normal(0, 1, len(idx)), 1e-4, None)
        runs.append(dict(td=torch.tensor(te[idx], dtype=torch.float32).reshape(-1, 1),
                         hd=torch.tensor(hm, dtype=torch.float32).reshape(-1, 1),
                         tau_s=te[idx], h_meas=hm, w=float(w), rpm=rpms[i], h=s.y[0], Pt=Pt(te), Et=Et(te),
                         hw_used=1.0, tr_used=1.0, raw_h0=1.0, raw_tmax=1.0, n_pts=len(idx), id=i))
    return runs

def ax0(a): a.tick_params(colors="#8a97a6"); a.grid(alpha=.22); return a
plt.rcParams.update({"figure.facecolor": "none", "axes.facecolor": "none", "axes.edgecolor": "#2a3340",
    "axes.labelcolor": "#c4cedd", "text.color": "#c4cedd", "xtick.color": "#8a97a6", "ytick.color": "#8a97a6",
    "axes.grid": True, "grid.color": "#1a212c", "axes.spines.top": False, "axes.spines.right": False})

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="kicker" style="margin-top:6px">controls</div>', unsafe_allow_html=True)
    source = st.radio("Data source", ["Synthetic", "Manual"], index=0,
                      help="Synthetic = built-in truth. Manual = your raw thickness data.")
    with st.expander("Synthetic ground truth & simulator", expanded=(source == "Synthetic")):
        PA = st.slider("Ψ_A", 0.1, 3.0, 1.2, 0.05); PD = st.slider("Ψ decay", 0.5, 6.0, 3.0, 0.1)
        EB = st.slider("Ẽ_B", 0.5, 6.0, 3.0, 0.1); ED = st.slider("Ẽ decay", 0.5, 6.0, 3.5, 0.1)
        rpm_a = st.slider("Run A · RPM", 1000, 6000, 3000, 100); rpm_b = st.slider("Run B · RPM", 1000, 6000, 4500, 100)
        n_meas = st.slider("Measurements / run", 4, 24, 8); noise = st.slider("Noise σ", 0.0, 0.10, 0.02, 0.005)
        seed = st.number_input("Seed", 0, 999, 42)
    with st.expander("Training", expanded=False):
        ea = st.slider("Phase A epochs", 100, 1500, 400, 50); eb = st.slider("Phase B epochs", 100, 1500, 500, 50)
        ec = st.slider("Phase C epochs", 100, 1500, 400, 50); ef = st.slider("Free-Ψ epochs", 200, 2000, 900, 100)
        lr = st.select_slider("Learning rate", [5e-4, 1e-3, 2e-3, 5e-3], value=1e-3)
        width = st.slider("Hidden width", 16, 64, 32, 8); layers = st.slider("Hidden layers", 2, 5, 3)
        wd = st.slider("W_data", 0.1, 5.0, 1.0, 0.1); wp = st.slider("W_physics", 0.1, 5.0, 1.0, 0.1)
        ns = st.slider("Multi-start restarts", 2, 10, 4)

# ── resolve runs + gate (before tabs, so every tab + the strip agree) ─────────
if source == "Synthetic":
    runs = demo_build(PA, PD, EB, ED, rpm_a, rpm_b, n_meas, noise, seed); has_truth = True
else:
    raw_runs = st.session_state.get('raw_runs', DEFAULT_RAW)
    auto_h = st.session_state.get('auto_h', True); auto_t = st.session_state.get('auto_t', True)
    gmax = max(float(r['t'].max()) for r in raw_runs)
    hw_manual = {}
    for r in raw_runs:
        rh0 = float(r['h'][int(np.argmin(r['t']))])
        hw_manual[r['id']] = rh0 if auto_h else float(st.session_state.get(f'hw_{r["id"]}', rh0))
    tr = gmax if auto_t else float(st.session_state.get('tr_manual', gmax))
    runs, rpm_ref = build_runs(raw_runs, auto_h, hw_manual, auto_t, tr); has_truth = False
issues, warns, diag = gate(runs)
level = 'block' if issues else ('warn' if warns else 'ok')
level_txt = {'ok': 'READY', 'warn': 'READY · NOTES', 'block': 'BLOCKED'}[level]
ws = sorted({round(r['w'], 2) for r in runs})

st.markdown(f"""
<div class="strip blk">
  <span class="k">source</span><span class="v">{source}</span><span class="sep">·</span>
  <span class="k">runs</span><span class="v">{len(runs)}</span><span class="sep">·</span>
  <span class="k">ω-scaling</span><span class="v">{ws}</span><span class="sep">·</span>
  <span class="k">normalization</span><span class="g-{level}">{level_txt}</span>
</div>""", unsafe_allow_html=True)

# ── tabs ──────────────────────────────────────────────────────────────────────
t_phys, t_data, t_train, t_res, t_man, t_mod = st.tabs(
    ["◐ Physics", "📡 Data", "🧠 Train", "📊 Results", "✎ Manual input", "◈ About"])

with t_phys:
    st.markdown(sec("01", "Forward simulator", "Integrate the ground-truth ODE at the sidebar spin speeds — intuition for what the PINN must invert."), unsafe_allow_html=True)
    te = np.linspace(0, 1, 500); c1, c2 = st.columns(2)
    with c1:
        f, a = plt.subplots(figsize=(6, 3.6), facecolor="none")
        for w, col in zip([rpm_a / rpm_a, rpm_b / rpm_a], ["#34d6cf", "#f0a93b"]):
            s = solve_ivp(lambda t, h, w=w: [-(w ** 2) * PA * np.exp(-PD * t) * h[0] ** 3 - EB * np.exp(-ED * t)], (0, 1), [1.0], t_eval=te, method='RK45')
            a.plot(te, s.y[0], color=col, lw=2.4, label=f"{int(w*rpm_a)} RPM")
        a.set_xlabel("τ"); a.set_ylabel("ĥ"); a.set_title("Thinning curves"); a.legend(frameon=False); ax0(a); st.pyplot(f)
    with c2:
        f, a = plt.subplots(figsize=(6, 3.6), facecolor="none")
        a.plot(te, PA * np.exp(-PD * te), color="#34d6cf", lw=2.4, label="Ψ(τ)")
        a.plot(te, EB * np.exp(-ED * te), color="#f0a93b", lw=2.4, label="Ẽ(τ)")
        a.set_xlabel("τ"); a.set_title("Hidden physics"); a.legend(frameon=False); ax0(a); st.pyplot(f)

with t_data:
    st.markdown(sec("02", "What the PINN sees", "Normalized sparse thickness per run (dots) plus unlabeled collocation ticks where the ODE is enforced."), unsafe_allow_html=True)
    cols = st.columns(min(len(runs), 2))
    for i in range(min(len(runs), 2)):
        with cols[i]:
            f, a = plt.subplots(figsize=(6, 3.6), facecolor="none")
            if has_truth: a.plot(runs[i].get('h') if runs[i].get('h') is not None else runs[i]['tau_s'], runs[i].get('h') if runs[i].get('h') is not None else runs[i]['h_meas'], color=["#34d6cf", "#f0a93b"][i], lw=2.2, alpha=.5, label="true ĥ")
            a.scatter(runs[i]['tau_s'], runs[i]['h_meas'], color=["#34d6cf", "#f0a93b"][i], s=46, zorder=5, label="data")
            a.scatter(np.sort(np.random.RandomState(i).uniform(0, 1, 200)), np.zeros(200), marker="|", color="#5a6675", s=50, label="collocation")
            a.set_xlabel("τ"); a.set_ylabel("ĥ"); a.set_title(f"Run {runs[i]['id']} · w={runs[i]['w']:.2f}"); a.legend(frameon=False, fontsize=8); ax0(a); st.pyplot(f)
    if not has_truth:
        st.caption(f"Normalization used — h_wet per run: " + ", ".join(f"run {r['id']}={r['hw_used']:.0f}" for r in runs) + f" · t_ref={runs[0]['tr_used']:.1f}.")

with t_train:
    st.markdown(sec("03", "Train the inverse model", "Constrained Ψ(τ)=A·exp(−d·τ) + shared Ẽ, then a free‑Ψ run as the unidentifiability diagnostic."), unsafe_allow_html=True)
    st.markdown(status_card(level, "Normalization gate — " + level_txt,
                            ([f"⛔ {x}" for x in issues] + [f"⚠ {x}" for x in warns]) or None), unsafe_allow_html=True)
    disabled = (level == 'block')
    if disabled: st.error("Training is blocked until the normalization issues above are resolved.")
    if st.button("🧠  Train inverse model", use_container_width=True, disabled=disabled):
        with st.spinner("Training constrained model, multi-start, and free‑Ψ diagnostic…"):
            param = train_parametric(runs, ea, eb, ec, lr, wd, wp, width, layers, seed)
            ms = multistart(param['hn'], runs, ns, 250, lr)
            free = train_free(runs, 300, ef, lr, width, layers, seed)
        st.session_state['train_res'] = dict(param=param, ms=ms, free=free)
        st.success("Training complete — see Results.")

with t_res:
    st.markdown(sec("04", "Inverse recovery", "Headline Ψ/Ẽ/h, the combined ODE term (the identifiable part), the information horizon, and a trust verdict."), unsafe_allow_html=True)
    tr = st.session_state.get('train_res')
    if tr is None:
        st.info("Press **Train** first.")
    else:
        param, ms, free = tr['param'], tr['ms'], tr['free']
        tau = np.linspace(0, 1, 300); tt = torch.tensor(tau, dtype=torch.float32).reshape(-1, 1)
        with torch.no_grad():
            Pp = param['psi'](tt).numpy().ravel(); Ep = param['en'](tt).numpy().ravel()
            hp = [param['hn'][i](tt, 1.0).numpy().ravel() for i in range(len(runs))]
        m1, m2, m3, m4 = st.columns(4)
        if has_truth:
            m1.metric("Ψ(τ) error", f"{rel(Pp, runs[0]['Pt']):.1f}%"); m2.metric("Ẽ(τ) error", f"{rel(Ep, runs[0]['Et']):.1f}%")
        else:
            m1.metric("Ψ(τ)", "no truth"); m2.metric("Ẽ(τ)", "no truth")
        h0e = rel(param['hn'][0](runs[0]['td'], 1.0).detach().numpy().ravel(), runs[0]['h_meas'])
        m3.metric("ĥ run A (fit)", f"{h0e:.1f}%")
        if len(runs) > 1:
            h1e = rel(param['hn'][1](runs[1]['td'], 1.0).detach().numpy().ravel(), runs[1]['h_meas']); m4.metric("ĥ run B (fit)", f"{h1e:.1f}%")
        else: m4.metric("ĥ run B", "—")
        c1, c2 = st.columns(2)
        with c1:
            f, a = plt.subplots(figsize=(6, 3.6), facecolor="none")
            if has_truth: a.plot(tau, runs[0]['Pt'], color="#34d6cf", lw=2.4, label="true Ψ"); a.plot(tau, runs[0]['Et'], color="#f0a93b", lw=2.4, label="true Ẽ")
            a.plot(tau, Pp, '--', color="#34d6cf", lw=2, label="pred Ψ"); a.plot(tau, Ep, '--', color="#f0a93b", lw=2, label="pred Ẽ")
            a.set_title("Shared Ψ & Ẽ"); a.legend(frameon=False, fontsize=8); ax0(a); st.pyplot(f)
        with c2:
            f, a = plt.subplots(figsize=(6, 3.6), facecolor="none")
            for i in range(len(runs)):
                cp = (runs[i]['w'] ** 2) * Pp * hp[i] ** 3 + Ep
                a.plot(tau, cp, '--', color=["#34d6cf", "#f0a93b"][i], lw=2, label=f"pred run{i}")
                if has_truth:
                    ct = (runs[i]['w'] ** 2) * runs[i]['Pt'] * runs[i]['h'] ** 3 + runs[i]['Et']; a.plot(tau, ct, color=["#34d6cf", "#f0a93b"][i], lw=2.4, alpha=.5, label=f"true run{i}")
            a.set_title("Combined ODE term (identifiable part)"); a.legend(frameon=False, fontsize=8); ax0(a); st.pyplot(f)
        hor = horizon_w(param['hn'], runs[0]['w'], runs[1]['w']) if len(runs) > 1 else None
        if hor:
            f, a = plt.subplots(figsize=(7, 3.4), facecolor="none")
            a.plot(hor['tau'], hor['c'], color="#e9edf2", lw=2); a.axvline(0.2, color="#ff6b6b", ls=':')
            a.set_title(f"2-run Ψ leverage c(τ) · {100*hor['frac_early']:.0f}% in τ<0.2"); ax0(a); st.pyplot(f)
        with torch.no_grad(): Pf = free['psi'](tt).numpy().ravel()
        f, a = plt.subplots(figsize=(7, 3.4), facecolor="none")
        if has_truth: a.plot(tau, runs[0]['Pt'], color="#34d6cf", lw=2.4, label="true Ψ")
        a.plot(tau, Pf, '--', color="#ff6b6b", lw=2, label="free Ψ (diagnostic)")
        a.set_title("Free‑Ψ diagnostic — fits the sum, hallucinates the split"); a.legend(frameon=False, fontsize=8); ax0(a); st.pyplot(f)
        v, arel, drel, note = verdict(ms, hor)
        st.markdown(status_card('warn' if drel > 0.5 else 'ok', "Trust verdict",
                                [f"{k}: {val}" for k, val in v.items()] + [note, f"multi-start Ψ_A spread/median={arel:.2f}, decay spread/median={drel:.2f}"]), unsafe_allow_html=True)

with t_man:
    st.markdown(sec("05", "Manual input & normalization", "Load raw dimensional data; the lab auto‑detects each run's wet thickness and checks the normalization live."), unsafe_allow_html=True)
    if source != "Manual":
        st.markdown('<div class="note">Switch <b>Data source → Manual</b> in the sidebar to load your own thickness data. The synthetic demo is already dimensionless (h_wet=1, τ∈[0,1]).</div>', unsafe_allow_html=True)
    else:
        st.markdown("**1 · Load raw data** (columns `run_id, t, h, rpm`; `rpm` optional). Edit the table, then press *Load & normalize*.", unsafe_allow_html=True)
        up = st.file_uploader("CSV file", type=["csv"], key="csvf")
        txt = st.text_area("or paste CSV", value="", height=150, key="paste", placeholder=EXAMPLE)
        ed = st.data_editor(EXAMPLE_DF, num_rows="dynamic", use_container_width=True, key="raw_ed", hide_index=True,
                            column_config={"run_id": st.column_config.NumberColumn("run_id", step=1),
                                           "rpm": st.column_config.NumberColumn("rpm", step=100)})
        if st.button("⬇  Load & normalize raw data", use_container_width=True):
            if up is not None: new = parse_text(up.getvalue().decode("utf-8-sig"))
            elif txt.strip(): new = parse_text(txt)
            else: new = parse_df(ed)
            if new:
                st.session_state['raw_runs'] = new
                for k in list(st.session_state.keys()):
                    if k.startswith('hw_') or k == 'tr_manual': st.session_state.pop(k, None)
                st.rerun()
            else: st.error("Could not read any rows — need at least columns `t` and `h`.")
        st.markdown("**2 · Normalization** (auto‑detect on by default; override per run if needed).", unsafe_allow_html=True)
        ca, cb = st.columns(2)
        with ca:
            auto_h = st.checkbox("Auto h_wet = each run's thickness at its earliest time", value=st.session_state.get('auto_h', True), key="auto_h")
            if not auto_h:
                for r in st.session_state.get('raw_runs', DEFAULT_RAW):
                    rh0 = float(r['h'][int(np.argmin(r['t']))])
                    st.number_input(f"h_wet run {r['id']}  (raw h@start ≈ {rh0:.0f})", value=float(st.session_state.get(f'hw_{r["id"]}', rh0)), key=f'hw_{r["id"]}')
        with cb:
            auto_t = st.checkbox("Auto t_ref = global max time", value=st.session_state.get('auto_t', True), key="auto_t")
            if not auto_t:
                st.number_input("t_ref", value=float(st.session_state.get('tr_manual', gmax)), key="tr_manual")
        st.markdown("**3 · Live normalization report**", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(diag, columns=["run", "rpm", "h_wet", "t_ref", "τ_min", "τ_max", "h̃@earliest", "pts"]),
                     use_container_width=True, hide_index=True)
        st.markdown(status_card(level, "Normalization gate — " + level_txt,
                                ([f"⛔ {x}" for x in issues] + [f"⚠ {x}" for x in warns]) or None), unsafe_allow_html=True)

with t_mod:
    st.markdown(sec("06", "About & honest limits", "What the gate checks, and what thickness data can and cannot tell you."), unsafe_allow_html=True)
    st.markdown("""
<div class="note"><b>The normalization gate</b> blocks training when, for any run, h̃ at the earliest
point is not ≈ 1 (your h_wet is wrong), τ leaves [0,1] (your t_ref is wrong), or thickness goes ≤ 0.
Each message names the run and the exact fix. Warnings (h̃>1.08, or a single ω‑scaling) never block.</div>

**What thickness recovers:** thickness (a few %), evaporation Ẽ (~10–20%), and the *combined* term
w²Ψĥ³+ (the identifiable part). With a constrained Ψ the amplitude/decaying shape come back right‑order.

**What it cannot:** the viscosity *decay rate*. Three independent diagnostics agree — the multi‑start
spread, the free‑Ψ mirror image, and the information horizon (≈95% of the two‑run Ψ leverage sits in
τ<0.2, where the film is still thick). Cleaner data won't fix it; the information isn't in terminal‑weighted
thickness. To see viscosity *dynamics*, sample τ<0.2 densely or add a viscosity‑sensitive observable.
""", unsafe_allow_html=True)
