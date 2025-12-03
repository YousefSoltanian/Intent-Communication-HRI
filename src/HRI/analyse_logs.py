#!/usr/bin/env python
# analyse_logs.py – full plot suite (stable flip, correct belief error)
# -------------------------------------------------------------------
import os, glob, pickle, math, pathlib, itertools
from collections import defaultdict
import numpy as np, numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from scipy.ndimage import gaussian_filter

# ─── configuration ───────────────────────────────────────────────
LOG_DIR = "logs"
FIG_DIR = pathlib.Path("figures"); FIG_DIR.mkdir(exist_ok=True)

SUBJECTS    = sorted({os.path.basename(f).split("_")[0]
                      for f in glob.glob(f"{LOG_DIR}/*.pkl")})
CONTROLLERS = ["blame", "npace", "influence"]
COLORS      = dict(blame="#1f77b4", npace="#ff7f0e", influence="#2ca02c")

NORM = 101; CI_Z = 1.96; ALPHA = .25; DPI = 400
BG    = "#001a33"; HEAT_BINS = 120

# ─── helpers ─────────────────────────────────────────────────────
def resample(t, v, n=NORM):
    t,v = np.asarray(t,float), np.asarray(v,float)
    if len(t) < 2:
        return np.full(n, np.nan if v.size==0 else v[0])
    return np.interp(np.linspace(0,1,n), t/t[-1], v)

def mean_ci(a):
    mu = np.nanmean(a,0)
    se = np.nanstd(a,0,ddof=1) / np.sqrt(np.sum(~np.isnan(a),0))
    return mu, CI_Z*se

def truth_low(idx):            # 1 if *low* pad is correct
    return 1 - (idx[1] if isinstance(idx,(list,tuple,np.ndarray)) else (idx & 1))

# ─── containers ─────────────────────────────────────────────────
metrics = defaultdict(lambda: defaultdict(list))
traj_succ, fuel_pts = defaultdict(list), []
glob_ratio, glob_dist, err_curves = defaultdict(list), defaultdict(list), defaultdict(list)
flip_rows = []

# ─── read logs ──────────────────────────────────────────────────
for subj, ctrl in itertools.product(SUBJECTS, CONTROLLERS):
    for tr in pickle.load(open(f"{LOG_DIR}/{subj}_{ctrl}.pkl","rb")):

        gx,gy = tr["goal_px"]; x0,y0 = tr["trajectory"][0]["state"][:2]
        d0    = math.hypot(x0-gx, y0-gy); low_t = truth_low(tr["goal_idx"])

        ts,dist,fh,fr,err,xs,ys = [],[],[],[],[],[],[]

        for st in tr["trajectory"]:
            ts.append(st["t"])
            x,y = st["state"][:2]; xs.append(x); ys.append(y)
            dist.append(math.hypot(x-gx,y-gy)/d0 if d0>1e-6 else 0.)
            fh.append(abs(st["F_h"])); fr.append(abs(st["delta_F_r"]))
            p = st["belief"]
            if p is None or np.isnan(p).all():
                err.append(np.nan)
            else:
                p_low = float(p[0] if np.ndim(p) else p)
                err.append(abs(p_low - low_t))

        # global curves
        denom = np.add(fh,fr)
        glob_ratio[ctrl].append(resample(ts, np.divide(fh, denom, out=np.zeros_like(denom,float), where=denom>0)))
        if tr["outcome"] == "success":
            glob_dist [ctrl].append(resample(ts, dist))
            traj_succ [ctrl].append((xs,ys))

        fuel_pts.append((np.trapz(denom, ts), int(tr["outcome"]=="success"), ctrl))
        if not np.all(np.isnan(err)):
            err_curves[ctrl].append(resample(ts, err))

        d_res = resample(ts, dist)
        metrics[subj][ctrl].append(dict(success=tr["outcome"]=="success",
                                        fuel=fuel_pts[-1][0],
                                        dist=d_res,
                                        aud =np.trapz(d_res, np.linspace(0,1,len(d_res)))*100,
                                        minpct=np.nanmin(d_res)*100))

        # flip metrics
        n=len(d_res); k=max(2,int(.15*n))
        err_arr=np.array(err,float)
        if n>=k+1 and not np.all(np.isnan(err_arr)):
            start, end = np.polyfit(range(k),d_res[:k],1)[0], np.polyfit(range(k),d_res[-k:],1)[0]
            flipped=int(start>0 and end<0)
            sw=np.where((err_arr[:-1]>0.5)&(err_arr[1:]<0.5))[0]
            t_sw=sw[0]/n*100 if sw.size else np.nan
            slope=np.nan
            if sw.size and sw[0]<n-2:
                i0,i1=sw[0],min(sw[0]+int(.1*n),n-1)
                slope=-(d_res[i1]-d_res[i0])/(0.1*n)
            flip_rows.append(dict(ctrl=ctrl,
                                  flipped=flipped,
                                  overshoot=np.nanmax(d_res)-d_res[0],
                                  t_switch=t_sw,
                                  slope=slope))

# ─── success-rate bar -------------------------------------------
fig,ax=plt.subplots(figsize=(6,4),dpi=DPI)
w=.25; ticks=np.arange(len(SUBJECTS))
for i,c in enumerate(CONTROLLERS):
    rate=[100*sum(m['success'] for m in metrics[s][c])/len(metrics[s][c]) for s in SUBJECTS]
    ax.bar(ticks+(i-1)*w,rate,w,color=COLORS[c],label=c.title())
ax.set(xticks=ticks,xticklabels=SUBJECTS,ylim=(0,100),
       ylabel="Success rate (%)",title="Landing success rate")
ax.legend(); fig.tight_layout(); fig.savefig(FIG_DIR/"success_rates.png"); plt.close(fig)

# ─── trajectory KDE ---------------------------------------------
fig,axes=plt.subplots(1,3,figsize=(15,5),dpi=DPI)
for ax,ctrl in zip(axes,CONTROLLERS):
    ax.set_facecolor(BG); ax.invert_yaxis()
    ax.set(title=f"{ctrl.title()} – successful",xlabel="x (px)",ylabel="y (px)")
    if not traj_succ[ctrl]: ax.text(.5,.5,"(none)",ha='center',va='center',color='w',transform=ax.transAxes); continue
    xs=np.concatenate([p[0] for p in traj_succ[ctrl]]); ys=np.concatenate([p[1] for p in traj_succ[ctrl]])
    H,xe,ye=np.histogram2d(xs,ys,bins=HEAT_BINS); H=gaussian_filter(H,2)
    ax.imshow(ma.masked_where(H==0,H).T,origin='lower',
              extent=[xe[0],xe[-1],ye[0],ye[-1]],
              cmap=plt.cm.magma,norm=mcolors.LogNorm(vmin=1,vmax=H.max()))
    for xs_t,ys_t in traj_succ[ctrl]:
        ax.plot(xs_t,ys_t,c='w',alpha=.35,lw=1.3); ax.scatter(xs_t[-1],ys_t[-1],s=14,c='w',alpha=.9)
fig.tight_layout(); fig.savefig(FIG_DIR/"trajectories_kde_success.png"); plt.close(fig)

# ─── global human-effort ----------------------------------------
fig,ax=plt.subplots(figsize=(6,4),dpi=DPI)
for c in CONTROLLERS:
    mu,ci=mean_ci(np.vstack(glob_ratio[c])); x=np.linspace(0,100,NORM)
    ax.plot(x,mu*100,c=COLORS[c],label=c.title())
    ax.fill_between(x,(mu-ci)*100,(mu+ci)*100,color=COLORS[c],alpha=ALPHA)
ax.set(xlabel="Progress (%)",ylabel="Human thrust (%)",ylim=(0,100),
       title="Human effort – all subjects"); ax.legend()
fig.tight_layout(); fig.savefig(FIG_DIR/"thrust_fraction_GLOBAL.png"); plt.close(fig)

# ─── global distance (successful) -------------------------------
fig,ax=plt.subplots(figsize=(6,4),dpi=DPI)
for c in CONTROLLERS:
    if not glob_dist[c]: continue
    mu,ci=mean_ci(np.vstack(glob_dist[c])); x=np.linspace(0,100,NORM)
    ax.plot(x,mu*100,c=COLORS[c],label=c.title())
    ax.fill_between(x,(mu-ci)*100,(mu+ci)*100,color=COLORS[c],alpha=ALPHA)
ax.set(xlabel="Progress (%)",ylabel="Remaining dist (%)",ylim=(0,100),
       title="Distance (successful) – all subjects"); ax.legend()
fig.tight_layout(); fig.savefig(FIG_DIR/"distance_success_GLOBAL.png"); plt.close(fig)

# ─── belief error curve -----------------------------------------
fig,ax=plt.subplots(figsize=(6,4),dpi=DPI)
for c in CONTROLLERS:
    if not err_curves[c]: continue
    mu,ci=mean_ci(np.vstack(err_curves[c])); x=np.linspace(0,100,NORM)
    ax.plot(x,mu,c=COLORS[c],label=c.title())
    ax.fill_between(x,mu-ci,mu+ci,color=COLORS[c],alpha=ALPHA)
ax.set(xlabel="Progress (%)",ylabel="Belief error",ylim=(0,1),
       title="Robot belief error"); ax.legend()
fig.tight_layout(); fig.savefig(FIG_DIR/"belief_error_global.png"); plt.close(fig)

# ─── flip metrics (A–D) -----------------------------------------
fig,axes=plt.subplots(4,1,figsize=(6,10),dpi=DPI,sharex=True)
axes[0].bar(range(3),
            [100*sum(r['flipped'] for r in flip_rows if r['ctrl']==c)/max(1,len([r for r in flip_rows if r['ctrl']==c]))
             for c in CONTROLLERS],color=[COLORS[c] for c in CONTROLLERS])
axes[0].set_xticks(range(3),[c.title() for c in CONTROLLERS]); axes[0].set(ylabel="% flips",title="A) diverge ➜ converge")

for i,c in enumerate(CONTROLLERS):
    overs=[r['overshoot'] for r in flip_rows if r['ctrl']==c]
    axes[1].scatter(i+np.random.uniform(-.07,.07,len(overs)),overs,10,c=COLORS[c],alpha=.7)
axes[1].set(ylabel="Peak overshoot",title="B) max error")

for i,c in enumerate(CONTROLLERS):
    t=[r['t_switch'] for r in flip_rows if r['ctrl']==c and not math.isnan(r['t_switch'])]
    if t: axes[2].scatter(i+np.random.uniform(-.07,.07,len(t)),t,10,c=COLORS[c],alpha=.7)
if not axes[2].collections: axes[2].text(.5,.5,"(none)",ha='center',va='center',transform=axes[2].transAxes)
axes[2].set(ylabel="Switch %",title="C) belief switch")

for i,c in enumerate(CONTROLLERS):
    slopes=[r['slope'] for r in flip_rows if r['ctrl']==c and not math.isnan(r['slope'])]
    if slopes:
        axes[3].violinplot(slopes,positions=[i],showmeans=True,widths=.6)
        axes[3].scatter(np.full(len(slopes),i),slopes,8,c=COLORS[c],alpha=.3)
if not axes[3].collections: axes[3].text(.5,.5,"(none)",ha='center',va='center',transform=axes[3].transAxes)
axes[3].set_xticks(range(3),[c.title() for c in CONTROLLERS]); axes[3].set(ylabel="Slope",title="D) correction speed")
fig.tight_layout(); fig.savefig(FIG_DIR/"flip_metrics.png"); plt.close(fig)

# ─── fuel vs success scatter ------------------------------------
fig,ax=plt.subplots(figsize=(6,4),dpi=DPI)
for c in CONTROLLERS:
    xs=[f for f,s,cc in fuel_pts if cc==c]; ys=[s+np.random.uniform(-.05,.05) for f,s,cc in fuel_pts if cc==c]
    ax.scatter(xs,ys,24,c=COLORS[c],alpha=.8,label=c.title())
ax.set_xlabel("Total fuel"); ax.set_yticks([0,1],["Fail","Success"]); ax.set_title("Fuel vs outcome"); ax.legend()
fig.tight_layout(); fig.savefig(FIG_DIR/"fuel_vs_success_all.png"); plt.close(fig)

# ─── per-subject AUD violin & min-dist box -----------------------
for subj in SUBJECTS:
    aud_data=[[m['aud'] for m in metrics[subj][c]] for c in CONTROLLERS]
    fig,ax=plt.subplots(figsize=(5,4),dpi=DPI)
    vp=ax.violinplot(aud_data,positions=[1,2,3],showmeans=True,widths=.8)
    for body,c in zip(vp['bodies'],CONTROLLERS): body.set_facecolor(COLORS[c]); body.set_alpha(.7)
    ax.set_xticks([1,2,3],[c.title() for c in CONTROLLERS]); ax.set_ylabel("Area-under-dist"); ax.set_title(f"{subj}: path efficiency")
    fig.tight_layout(); fig.savefig(FIG_DIR/f"aud_violin_{subj}.png"); plt.close(fig)

    md_data=[[m['minpct'] for m in metrics[subj][c]] for c in CONTROLLERS]
    fig,ax=plt.subplots(figsize=(5,4),dpi=DPI)
    bp=ax.boxplot(md_data,positions=[1,2,3],patch_artist=True,widths=.6)
    for patch,c in zip(bp['boxes'],CONTROLLERS): patch.set_facecolor(COLORS[c]); patch.set_alpha(.6)
    ax.set_xticks([1,2,3],[c.title() for c in CONTROLLERS]); ax.set_ylabel("Min dist (%)"); ax.set_title(f"{subj}: closest approach")
    fig.tight_layout(); fig.savefig(FIG_DIR/f"min_dist_box_{subj}.png"); plt.close(fig)

print("✓ Figures saved →", FIG_DIR.resolve())
