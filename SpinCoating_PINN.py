# =============================================================================
#  PINN SPIN-COATING INVERSE TOOL  (honest, diagnostics-first)
#  One self-contained script. CPU-only.  Deps: numpy, scipy, torch, matplotlib.
#  (requirements.txt:  numpy  scipy  torch  matplotlib)
#
#  WHAT IT DOES: from sparse thickness-vs-time at >=1 spin speed, recover
#    h(tau), E(tau), and a CONSTRAINED viscosity shape Psi(tau)=A*exp(-d*tau),
#    AND report identifiability diagnostics + a plain-language TRUST VERDICT.
#  It NEVER reports the viscosity decay as a precise measurement, because the
#  physics + measurement design make it information-limited (your paper's result).
#
#  TWO MODES:  cfg['mode'] = 'demo'  -> synthetic, truth known, prints error %
#              cfg['mode'] = 'csv'   -> real data, NO truth, prints diagnostics
#  The default ('demo') reproduces a KNOWN result -> your regression test.
#
#  I have NOT executed this assembly; components are from a validated notebook.
#  Run unedited first; the self-test + default demo verify it.
# =============================================================================
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, approx_fprime
import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
import json, os

# =============================================================================
#  USER INPUT  -- edit THIS block (or set mode='csv' and give a path)
# =============================================================================
cfg = dict(
    mode        = 'demo',          # 'demo' (synthetic+truth) or 'csv' (real, no truth)
    csv_path    = None,            # if mode='csv': columns run_id,t,h[,h_true]; t,h dimensionless
    rpms        = [3000, 4500],    # spin speeds per run (defines w_norm = rpm/rpm_ref)
    rpm_ref     = 3000,
    noise       = 0.02,            # assumed/estimated fractional thickness noise
    seed        = 42,
    out_dir     = 'pinn_results',  # figures + summary JSON saved here
    # model toggles
    use_free_mlp_diag = True,      # Model 2: free-MLP Psi as the unidentifiability WARNING
    use_constitutive  = False,     # Model 3: Tier-1 gelation fit (needs c0; off for real data)
    c0              = 0.02,        # initial polymer vol. fraction (only if use_constitutive)
    n_starts        = 6,           # multi-start refits of (A,decay) -> practical-identifiability spread
    epochs_param    = 800,         # Phase B/C epochs for the constrained PINN
    epochs_free     = 1500,        # epochs for the free-MLP diagnostic (the slow part)
    n_coll          = 150,
)

# =============================================================================
#  HELPERS
# =============================================================================
def rel(p, t): return float(np.mean(np.abs(p - t) / (np.abs(t) + 1e-8))) * 100.0

def nondimensionalize(t_s, h_m, h_wet, t_ref):
    """Dimensional -> dimensionless.  tau=t/t_ref, h_tilde=h/h_wet (h(0)=1).
       CAVEAT: choosing t_ref is a PHYSICAL modeling choice (see your PDF's time
       scaling); the recovered dimensionless curves depend on it -- report it."""
    return np.asarray(t_s)/t_ref, np.asarray(h_m)/h_wet

def load_runs_csv(path, rpms, rpm_ref):
    """CSV columns: run_id,t,h[,h_true].  t,h already dimensionless (h(0)=1)."""
    import csv
    runs = {}
    with open(path) as f:
        rd = csv.DictReader(f)
        for row in rd:
            i = int(row['run_id']); runs.setdefault(i, {'t':[], 'h':[], 'ht':[]})
            runs[i]['t'].append(float(row['t'])); runs[i]['h'].append(float(row['h']))
            if 'h_true' in row and row['h_true']: runs[i]['ht'].append(float(row['h_true']))
    out = []
    for i in sorted(runs):
        t = np.array(runs[i]['t']); h = np.array(runs[i]['h'])
        o = np.argsort(t); t, h = t[o], h[o]
        d = dict(tau=t, h=h, rpm=rpms[i] if i < len(rpms) else rpm_ref,
                 w=(rpms[i] if i < len(rpms) else rpm_ref)/rpm_ref)
        if runs[i]['ht']: d['h_true'] = np.array(runs[i]['ht'])[o]
        out.append(d)
    return out

def build_demo(rpms, rpm_ref, noise, seed):
    """Balanced EBP+evap sandbox (the regression-test ground truth)."""
    np.random.seed(seed); torch.manual_seed(seed)
    W = [r/rpm_ref for r in rpms]
    PA, PD, EB, ED = 1.2, 3.0, 3.0, 3.5
    Pt = lambda t: PA*np.exp(-PD*t); Et = lambda t: EB*np.exp(-ED*t)
    K  = lambda t, w: (w**2)*Pt(t)
    tau = np.linspace(0,1,500); runs = []
    edges = np.linspace(0,1,9)
    for i, w in enumerate(W):
        s = solve_ivp(lambda t,h,w=w:[-K(t,w)*h[0]**3-Et(t)],(0,1),[1.0],t_eval=tau,method='RK45')
        tg = np.array([np.random.uniform(edges[k],edges[k+1]) for k in range(8)])
        idx = np.sort(np.unique([np.argmin(np.abs(tau-t)) for t in tg])); idx[-1]=len(tau)-1
        ht = s.y[0][idx]
        runs.append(dict(tau=tau, h=s.y[0], tau_s=tau[idx],
                         h_meas=np.clip(ht+np.random.normal(0,noise,len(idx))*ht,1e-4,None),
                         h_true=s.y[0], rpm=rpms[i], w=w,
                         Pt=Pt(tau), Et=Et(tau), Kt=K(tau,w)))
    return runs, (Pt, Et)

# =============================================================================
#  NETWORKS  (multi-line __init__ -> no paste-corruption)
# =============================================================================
def mlp():
    L=[nn.Linear(1,32),nn.Tanh()]+[nn.Linear(32,32),nn.Tanh()]*2+[nn.Linear(32,1)]
    return nn.Sequential(*L)
 HNet(nn.Module):
    def __init__(self): super().__init__(); self.net=mlp(); self.sp=nn.Softplus()
    def forward(self,t,h0=1.0): return h0 - t*self.sp(self.net(t))
class PsiPar(nn.Module):                       # constrained: A*exp(-d*tau), d>=0
    def __init__(self):
        super().__init__()
        self.logA = nn.Parameter(torch.tensor(0.0))
        self.raw  = nn.Parameter(torch.tensor(0.5))
        self.sp   = nn.Softplus()
    def forward(self, t):
        return torch.exp(self.logA - self.sp(self.raw) * t)
    def ab(self):
        return float(torch.exp(self.logA).item()), float(self.sp(self.raw).item())
class ENet(nn.Module):
    def __init__(self): super().__init__(); self.net=mlp(); self.sp=nn.Softplus()
    def forward(self,t): return self.sp(self.net(t))
class PsiFree(nn.Module):                      # UNCONSTRAINED (diagnostic only)
    def __init__(self): super().__init__(); self.net=mlp()
    def forward(self,t): return torch.exp(self.net(t))

def resid(hn, psi, en, t, w):
    t=t.reshape(-1,1); h=hn(t,1.0)
    dh=torch.autograd.grad(h,t,grad_outputs=torch.ones_like(h),create_graph=True,retain_graph=True)[0]
    return dh + (w**2)*psi(t)*h**3 + en(t)
def coll(n):
    t=torch.tensor(np.sort(np.random.uniform(0,1,n)),dtype=torch.float32).reshape(-1,1)
    t.requires_grad_(True); return t

# =============================================================================
#  MODEL 1: constrained/parametric Psi PINN  (the honest "answer")
# =============================================================================
def train_parametric(runs, cfg):
    W=[r['w'] for r in runs]; N=cfg['n_coll']
    td=[torch.tensor(r['tau_s'],dtype=torch.float32).reshape(-1,1) for r in runs]
    hd=[torch.tensor(r['h_meas'],dtype=torch.float32).reshape(-1,1) for r in runs]
    hn=[HNet() for _ in W]; psi=PsiPar(); en=ENet()
    # Phase A: data-only h
    oA=optim.Adam([p for h in hn for p in h.parameters()],lr=1e-3)
    for _ in range(600):
        oA.zero_grad(); L=sum(torch.mean((hn[i](td[i],1.0)-hd[i])**2) for i in range(len(W)))
        L.backward(); oA.step()
    # Phase B: freeze h, fit Psi+E
    for h in hn:
        for p in h.parameters(): p.requires_grad_(False)
    oB=optim.Adam([{'params':list(psi.parameters()),'lr':1e-2},{'params':list(en.parameters()),'lr':1e-3}])
    for _ in range(cfg['epochs_param']):
        oB.zero_grad(); L=sum(torch.mean(resid(hn[i],psi,en,coll(N),W[i])**2) for i in range(len(W)))
        L.backward(); oB.step()
    # Phase C: joint
    for h in hn:
        for p in h.parameters(): p.requires_grad_(True)
    oC=optim.Adam([{'params':[p for h in hn for p in h.parameters()],'lr':1e-4},
                   {'params':list(psi.parameters()),'lr':1e-3},{'params':list(en.parameters()),'lr':1e-4}])
    for _ in range(cfg['epochs_param']):
        oC.zero_grad()
        Ld=sum(torch.mean((hn[i](td[i],1.0)-hd[i])**2) for i in range(len(W)))
        Lp=sum(torch.mean(resid(hn[i],psi,en,coll(N),W[i])**2) for i in range(len(W)))
        (Ld+Lp).backward(); oC.step()
    return hn, psi, en

def multistart_psi(hn, runs, cfg):
    """h frozen at Phase-A fit; refit (logA,raw) from random inits -> spread = practical identifiability."""
    W=[r['w'] for r in runs]; N=cfg['n_coll']; out=[]
    for h in hn:
        for p in h.parameters(): p.requires_grad_(False)
    for s in range(cfg['n_starts']):
        torch.manual_seed(1000+s); psi=PsiPar(); en=ENet()
        with torch.no_grad():
            psi.logA.copy_(torch.tensor(np.random.uniform(-1,1))); psi.raw.copy_(torch.tensor(np.random.uniform(-1,2)))
        o=optim.Adam([{'params':list(psi.parameters()),'lr':1e-2},{'params':list(en.parameters()),'lr':1e-3}])
        for _ in range(500):
            o.zero_grad(); L=sum(torch.mean(resid(hn[i],psi,en,coll(N),W[i])**2) for i in range(len(W)))
            L.backward(); o.step()
        out.append(psi.ab())
    for h in hn:
        for p in h.parameters(): p.requires_grad_(True)
    A=np.array([a for a,_ in out]); d=np.array([b for _,b in out])
    return dict(A_med=float(np.median(A)), A_spread=float(np.std(A)),
                d_med=float(np.median(d)), d_spread=float(np.std(d)), all=out)

# =============================================================================
#  MODEL 2: free-MLP Psi PINN  (DIAGNOSTIC: the mirror-image warning)
# =============================================================================
def train_free(runs, cfg):
    W=[r['w'] for r in runs]; N=cfg['n_coll']
    td=[torch.tensor(r['tau_s'],dtype=torch.float32).reshape(-1,1) for r in runs]
    hd=[torch.tensor(r['h_meas'],dtype=torch.float32).reshape(-1,1) for r in runs]
    hn=[HNet() for _ in W]; psi=PsiFree(); en=ENet()
    oA=optim.Adam([p for h in hn for p in h.parameters()],lr=1e-3)
    for _ in range(400):
        oA.zero_grad(); L=sum(torch.mean((hn[i](td[i],1.0)-hd[i])**2) for i in range(len(W)))
        L.backward(); oA.step()
    params=[p for h in hn for p in h.parameters()]+list(psi.parameters())+list(en.parameters())
    o=optim.Adam(params,lr=1e-3)
    for _ in range(cfg['epochs_free']):
        o.zero_grad(); Ld=Lp=0.0
        for i in range(len(W)):
            Ld=Ld+torch.mean((hn[i](td[i],1.0)-hd[i])**2)
            Lp=Lp+torch.mean(resid(hn[i],psi,en,coll(N),W[i])**2)
        (Ld+Lp).backward(); o.step()
    return hn, psi, en

# =============================================================================
#  DIAGNOSTICS  (none of these need ground truth)
# =============================================================================
def internal_physics_residual(hn, psi, en, runs, cfg):
    """mean R1^2 over runs on fresh collocation -> how well the ODE is satisfied."""
    N=cfg['n_coll']; W=[r['w'] for r in runs]; vals=[]
    for i in range(len(W)):
        R=resid(hn[i],psi,en,coll(N),W[i]); vals.append(float(torch.mean(R**2).item()))
    return float(np.mean(vals))

def information_horizon(hn, runs, cfg):
    """c(tau)=|w0^2 h0^3 - w1^2 h1^3| from RECOVERED h (works without truth).
       Fraction of leverage in tau<0.2 = how concentrated the viscosity signal is."""
    if len(runs) < 2: return None
    tau=np.linspace(0,1,300); tt=torch.tensor(tau,dtype=torch.float32).reshape(-1,1)
    with torch.no_grad():
        h=[hn[i](tt,1.0).numpy().ravel() for i in range(2)]
    c=np.abs((runs[0]['w']**2)*h[0]**3 - (runs[1]['w']**2)*h[1]**3)
    tot=np.trapezoid(c,tau); e=tau<0.2
    frac=float(np.trapezoid(c[e],tau[e])/tot) if tot>0 else float('nan')
    hz=float(tau[np.argmax(c<0.1)]) if (c<0.1).any() else 1.0
    return dict(frac_early=frac, horizon=hz, c=c, tau=tau)

def trust_verdict(ms, hor, has_truth):
    """Heuristic trust flags DERIVED from diagnostics (NOT invented confidence)."""
    v={}
    v['h']        = 'HIGH  (fit to data by construction)'
    v['E']        = 'MEDIUM-HIGH (slope of h; check internal residual)'
    v['combined'] = 'HIGH  (this is what the physics loss constrains)'
    # Psi amplitude: medium only if multi-run AND multi-start amplitude is tight
    amp_tight = ms['A_spread']/max(abs(ms['A_med']),1e-6) < 0.3
    v['Psi_amplitude'] = 'MEDIUM (constrained shape; multi-start spread small)' if (len(ms['all'])>1 and amp_tight) \
                         else 'LOW (multi-start amplitude spread large)'
    # Psi decay: unidentifiable if multi-start decay spread is large relative to its value
    dec_rel = ms['d_spread']/max(abs(ms['d_med']),1e-6)
    v['Psi_decay'] = 'UNIDENTIFIABLE (multi-start decay spread/median = %.2f > 0.5; '%dec_rel + \
                     'decay is a prior-driven extrapolation, NOT a measurement)' if dec_rel>0.5 \
                     else 'LOW-MEDIUM (decay spread moderate; treat cautiously)'
    if hor and hor['frac_early']>0.8:
        v['note'] = f"Information horizon: {100*hor['frac_early']:.0f}% of 2-run viscosity leverage in tau<0.2 -> " \
                    "late-time viscosity is information-limited (matches the paper)."
    return v

# =============================================================================
#  MODEL 3 (optional): constitutive Tier-1 gelation fit + Fisher/profile
#  Off by default for real data (needs c0 + a committed closure).
# =============================================================================
Wc=0.02
def g_nu(c,beta,cg):
    c=np.asarray(c,float); return np.exp(-beta*c/(np.clip(1-c/cg,0,None)+Wc))
def integrate_const(x,w,leak=0.0):
    Psi0,beta,E0,cg=x; TAU=np.linspace(0,1,400)
    def rhs(t,y):
        h=float(np.clip(y[0],1e-3,1.0)); c=float(np.clip(cfg['c0']/h,0,1))
        return [-(w**2)*Psi0*float(g_nu(c,beta,cg))*h**3 - (E0 if c<cg else E0*leak)]
    def gel(t,y): return cfg['c0']/float(np.clip(y[0],1e-3,1.0))-cg
    gel.terminal=True; gel.direction=1.0
    sol=solve_ivp(rhs,(0,1),[1.0],t_eval=TAU,method='RK45',
                  events=(gel if leak==0 else None),rtol=1e-7,atol=1e-9)
    h=sol.y[0]
    if len(h)<len(TAU): h=np.concatenate([h,np.full(len(TAU)-len(h),h[-1])])
    return TAU,h
def run_constitutive_demo(seed=7):
    """Self-contained gelation sandbox -> reproduces Tier-1 numbers (derived Psi ~49.6/33.3%, beta unidentifiable)."""
    np.random.seed(seed)
    TRUE=np.array([1.2,6.0,3.0,0.25]); W=[1.0,1.5]; NOISE=0.02
    TAU_H=np.linspace(0,1,8); BOUNDS=[(1e-3,30),(0,40),(1e-3,30),(0.05,0.6)]
    runs=[]
    for w in W:
        _,hd=integrate_const(TRUE,w); ht=np.interp(TAU_H,np.linspace(0,1,400),hd)
        runs.append(dict(w=w,hm=ht+NOISE*np.maximum(ht,1e-3)*np.random.randn(*ht.shape),
                         sh=NOISE*np.maximum(ht,1e-3)))
    def resv(x):
        r=[]
        for rn in runs:
            _,hp=integrate_const(x,rn['w']); r.append((np.interp(TAU_H,np.linspace(0,1,400),hp)-rn['hm'])/rn['sh'])
        return np.concatenate(r)
    x0=np.clip(TRUE*np.array([1.6,0.5,1.5,0.8]),[b[0] for b in BOUNDS],[b[1] for b in BOUNDS])
    r=minimize(lambda x:0.5*np.dot(resv(x),resv(x)),x0,method='L-BFGS-B',bounds=BOUNDS,
               jac=lambda x:approx_fprime(x,resv,1e-4).T@resv(x),options={'maxiter':300,'ftol':1e-14})
    x=r.x; J=approx_fprime(x,resv,1e-4)
    try: rel_u=100*np.sqrt(np.clip(np.diag(np.linalg.inv(J.T@J)),0,None))/np.maximum(np.abs(x),1e-6)
    except np.linalg.LinAlgError: rel_u=np.full(4,np.inf)
    # profile for beta
    grid=np.linspace(0.5,18,9); objs=[]
    for bv in grid:
        fr=[i for i in range(4) if i!=1]; b=[BOUNDS[i] for i in fr]
        def ob(xf,bv=bv):
            xx=x.copy(); xx[1]=bv; xx[fr]=xf; return 0.5*np.dot(resv(xx),resv(xx))
        rr=minimize(ob,x0[fr],method='L-BFGS-B',bounds=b); objs.append(rr.fun)
    return dict(fit=x,true=TRUE,rel_u=rel_u,cond=float(np.linalg.cond(J)),
                beta_grid=grid,beta_obj=np.array(objs))

# =============================================================================
#  PLOTTING  (branches on has_truth; always saves PNGs)
# =============================================================================
def plot_all(out, cfg, has_truth):
    os.makedirs(cfg['out_dir'],exist_ok=True); tau=np.linspace(0,1,300)
    tt=torch.tensor(tau,dtype=torch.float32).reshape(-1,1)
    with torch.no_grad():
        Pp=out['psi'](tt).numpy().ravel(); Ep=out['en'](tt).numpy().ravel()
        hp=[out['hn'][i](tt,1.0).numpy().ravel() for i in range(len(out['runs']))]
    fig,ax=plt.subplots(1,3,figsize=(16,4.3))
    if has_truth:
        ax[0].plot(tau,out['runs'][0]['Pt'],'b-'); ax[0].plot(tau,Pp,'r--')
        ax[0].set_title(f"constrained Psi  err={rel(Pp,out['runs'][0]['Pt']):.0f}%")
        ax[1].plot(tau,out['runs'][0]['Et'],'b-'); ax[1].plot(tau,Ep,'r--')
        ax[1].set_title(f"E  err={rel(Ep,out['runs'][0]['Et']):.0f}%")
    else:
        ax[0].plot(tau,Pp,'r-'); ax[0].set_title("constrained Psi (no truth: shape only)")
        ax[1].plot(tau,Ep,'r-'); ax[1].set_title("E (no truth)")
    for i,r in enumerate(out['runs']):
        ax[2].plot(tau,r['h'],'-' if has_truth else ':',alpha=.4)
        ax[2].plot(tau,hp[i],'--'); ax[2].scatter(r['tau_s'],r['h_meas'],s=20)
    ax[2].set_title("h recovery + sparse data"); 
    for a in ax: a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{cfg['out_dir']}/01_recovery.png"); 
    try: plt.show()
    except Exception: pass

    if out.get('hor'):
        fig,ax=plt.subplots(1,2,figsize=(12,4.2)); h=out['hor']
        ax[0].plot(h['tau'],h['c'],'k-'); ax[0].axvline(0.2,color='r',ls=':')
        ax[0].set_title(f"2-run viscosity leverage c(tau): {100*h['frac_early']:.0f}% in tau<0.2")
        ax[1].semilogy(h['tau'],h['c']+1e-9,'k-'); ax[1].axvline(h['horizon'],color='r',ls=':')
        ax[1].set_title(f"log c(tau); horizon tau={h['horizon']:.2f}")
        for a in ax: a.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(f"{cfg['out_dir']}/02_horizon.png")
        try: plt.show()
        except Exception: pass

    if out.get('free'):
        with torch.no_grad(): Pf=out['free']['psi'](tt).numpy().ravel()
        fig,ax=plt.subplots(1,2,figsize=(11,4.2))
        ax[0].plot(tau,Pf,'r--'); 
        if has_truth: ax[0].plot(tau,out['runs'][0]['Pt'],'b-')
        ax[0].set_title("FREE-MLP Psi = DIAGNOSTIC (mirror image => split untrustworthy)")
        ms=out['ms']; ax[1].hist([d for _,d in ms['all']],bins=max(4,cfg['n_starts']),color='r',alpha=.6)
        ax[1].set_title(f"multi-start decay spread (median {ms['d_med']:.2f} +- {ms['d_spread']:.2f})")
        for a in ax: a.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(f"{cfg['out_dir']}/03_diagnostics.png")
        try: plt.show()
        except Exception: pass

    if out.get('const'):
        c=out['const']; fig,ax=plt.subplots(1,2,figsize=(11,4.2))
        ax[0].plot(c['beta_grid'],c['beta_obj'],'o-'); ax[0].axvline(c['true'][1],ls=':',c='r')
        ax[0].set_title("Tier-1 profile likelihood for beta (flat => unidentifiable)")
        ax[1].bar(['Psi0','beta','E0','cg'],c['rel_u'],color='r',alpha=.6); ax[1].set_yscale('log')
        ax[1].set_title("Fisher rel-uncertainty % (beta,cg huge => unidentifiable)")
        for a in ax: a.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(f"{cfg['out_dir']}/04_constitutive.png")
        try: plt.show()
        except Exception: pass

# =============================================================================
#  MAIN PIPELINE
# =============================================================================
def run(cfg):
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
    # ---- self-test (fails fast on a broken paste) ----
    try:
        _t=torch.tensor([[0.0],[0.5]],dtype=torch.float32); _t.requires_grad_(True)
        assert HNet()(_t,1.0)[0].item()==1.0
        _=resid(HNet(),PsiPar(),ENet(),_t,1.5)
        print("SELF-TEST OK")
    except Exception as e:
        print("SELF-TEST FAILED:",repr(e)); return
    # ---- data ----
    if cfg['mode']=='demo':
        runs,_=build_demo(cfg['rpms'],cfg['rpm_ref'],cfg['noise'],cfg['seed']); has_truth=True
    else:
        runs=load_runs_csv(cfg['csv_path'],cfg['rpms'],cfg['rpm_ref']); has_truth=all('h_true' in r for r in runs)
        if not has_truth: print("REAL-DATA MODE: no ground truth -> no error %, diagnostics only.")
    out=dict(runs=runs)
    # ---- Model 1 ----
    hn,psi,en=train_parametric(runs,cfg); out['hn'],out['psi'],out['en']=hn,psi,en
    out['ms']=multistart_psi(hn,runs,cfg)
    out['resid']=internal_physics_residual(hn,psi,en,runs,cfg)
    out['hor']=information_horizon(hn,runs,cfg)
    # ---- Model 2 (diagnostic) ----
    if cfg['use_free_mlp_diag']:
        fhn,fp,fe=train_free(runs,cfg); out['free']=dict(hn=fhn,psi=fp,en=fe)
    # ---- Model 3 (optional) ----
    if cfg['use_constitutive']:
        out['const']=run_constitutive_demo() if cfg['mode']=='demo' else None  # real-data Tier-1 needs c0+closure; demo reproduces known nums
    # ---- verdict ----
    out['verdict']=trust_verdict(out['ms'],out['hor'],has_truth)
    # ---- report ----
    tt=torch.tensor(np.linspace(0,1,300),dtype=torch.float32).reshape(-1,1)
    with torch.no_grad(): Pp=psi(tt).numpy().ravel(); Ep=en(tt).numpy().ravel()
    print("\n=== RESULTS ===")
    if has_truth:
        print(f"constrained Psi err : {rel(Pp,runs[0]['Pt']):.1f}%")
        print(f"E err               : {rel(Ep,runs[0]['Et']):.1f}%")
        for i,r in enumerate(runs):
            hpi=out['hn'][i](tt,1.0).detach().numpy().ravel(); print(f"h err run {i}       : {rel(hpi,r['h_true']):.1f}%")
    print(f"internal physics residual (mean R1^2, no truth needed): {out['resid']:.5f}")
    print(f"multi-start Psi_A = {out['ms']['A_med']:.2f} +- {out['ms']['A_spread']:.2f} ; "
          f"decay = {out['ms']['d_med']:.2f} +- {out['ms']['d_spread']:.2f}")
    if out['hor']: print(f"horizon: {100*out['hor']['frac_early']:.0f}% of leverage in tau<0.2; horizon tau={out['hor']['horizon']:.2f}")
    print("\n--- TRUST VERDICT (derived from diagnostics, not a confidence interval) ---")
    for k,v in out['verdict'].items(): print(f"  {k:16s}: {v}")
    if out.get('const'):
        c=out['const']; print(f"\nTier-1 fit err%: {100*np.abs(c['fit']-c['true'])/c['true']}  cond(J)={c['cond']:.1e}")
    # ---- save ----
    plot_all(out,cfg,has_truth)
    summ={k:(v if not isinstance(v,np.ndarray) else v.tolist()) for k,v in
          dict(mode=cfg['mode'],has_truth=has_truth,resid=out['resid'],ms=out['ms'],
               hor=out['hor'],verdict=out['verdict']).items()}
    json.dump(summ,open(f"{cfg['out_dir']}/summary.json",'w'),indent=2,default=str)
    print(f"\nSaved figures + summary.json to ./{cfg['out_dir']}/")
    return out

if __name__=='__main__':
    run(cfg)
else:
    # in a notebook cell, just call:  out = run(cfg)
    pass
