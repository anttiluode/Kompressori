#!/usr/bin/env python3
"""
PhiWorld2 partial-field completion probe.

Tests how much of an evolving PhiWorld field must remain visible/locked before
the hidden remainder follows the full reference trajectory.

Two modes:
  1) initial_only:
     Keep only a fraction of (phi, phi_old), zero the rest, then run freely.
  2) trajectory_clamp:
     Keep only that same spatial fraction continuously synchronized to the
     reference trajectory while measuring reconstruction ONLY on hidden cells.

This is a dynamical pattern-completion / synchronization experiment, not a test
of the holographic principle.

Outputs:
  completion_results.csv
  completion_summary.json
  completion_curves.png
  reference_and_masks.png
  example_reconstructions.png
"""

from __future__ import annotations
import argparse, csv, json
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class Params:
    dt: float = 0.08
    damping: float = 0.001
    base_c_sq: float = 1.0
    tension_factor: float = 5.0
    potential_lin: float = 1.0
    potential_cub: float = 0.2
    biharmonic_gamma: float = 0.02


def lap(x):
    return (
        np.roll(x,1,0)+np.roll(x,-1,0)+
        np.roll(x,1,1)+np.roll(x,-1,1)-4*x
    )


def accel(phi, p):
    L = lap(phi)
    B = lap(L)
    c2 = p.base_c_sq/(1+p.tension_factor*phi*phi+1e-9)
    return c2*L + p.potential_lin*phi - p.potential_cub*phi**3 - p.biharmonic_gamma*B


def step(phi, old, p):
    vel = phi-old
    new = phi + (1-p.damping*p.dt)*vel + p.dt**2*accel(phi,p)
    return new, phi


def initial_gaussian(n):
    yy,xx=np.mgrid[:n,:n]
    c=n//2
    r=n/15
    phi=2*np.exp(-((xx-c)**2+(yy-c)**2)/(2*r*r))
    return phi.astype(float), phi.astype(float).copy()


def build_reference(n,p,target_steps,horizon):
    phi,old=initial_gaussian(n)
    for _ in range(target_steps):
        phi,old=step(phi,old,p)
    target, target_old = phi.copy(), old.copy()
    ref=[target.copy()]
    a,b=target.copy(),target_old.copy()
    for _ in range(horizon):
        a,b=step(a,b,p)
        ref.append(a.copy())
    return target,target_old,np.asarray(ref)


def exact_mask(score,fraction):
    flat=score.ravel()
    k=max(1,min(flat.size-1,int(round(fraction*flat.size))))
    idx=np.argpartition(flat,k-1)[:k]
    m=np.zeros(flat.size,bool); m[idx]=True
    return m.reshape(score.shape)


def make_mask(kind,n,fraction,rng):
    yy,xx=np.mgrid[:n,:n]
    c=(n-1)/2
    if kind=="random":
        k=max(1,min(n*n-1,int(round(fraction*n*n))))
        idx=rng.choice(n*n,k,replace=False)
        m=np.zeros(n*n,bool); m[idx]=True
        return m.reshape(n,n)
    if kind=="center":
        return exact_mask((xx-c)**2+(yy-c)**2,fraction)
    if kind=="boundary":
        s=np.minimum.reduce([xx,yy,n-1-xx,n-1-yy]).astype(float)
        s += rng.uniform(0,1e-6,s.shape)
        return exact_mask(s,fraction)
    if kind=="ring":
        r=np.sqrt((xx-c)**2+(yy-c)**2)
        s=np.abs(r-0.27*n)+rng.uniform(0,1e-6,r.shape)
        return exact_mask(s,fraction)
    raise ValueError(kind)


def metrics(pred,truth,mask):
    h=~mask
    a=pred[h].astype(float); b=truth[h].astype(float)
    mse=float(np.mean((a-b)**2))
    energy=float(np.mean(b*b))+1e-15
    a0=a-a.mean(); b0=b-b.mean()
    den=float(np.linalg.norm(a0)*np.linalg.norm(b0))
    corr=float(a0@b0/den) if den>1e-15 else 0.0
    den2=float(np.linalg.norm(a)*np.linalg.norm(b))
    cosine=float(a@b/den2) if den2>1e-15 else 0.0
    return {
        "hidden_mse":mse,
        "hidden_nrmse":float(np.sqrt(mse/energy)),
        "hidden_corr":corr,
        "hidden_cosine":cosine,
        "hidden_skill":float(1-mse/energy),
    }


def reconstruct(target,target_old,ref,mask,p,mode,noise,rng):
    phi=np.zeros_like(target); old=np.zeros_like(target_old)
    phi[mask]=target[mask]; old[mask]=target_old[mask]
    if noise>0:
        h=~mask
        phi[h]=rng.normal(0,noise,h.sum())
        old[h]=rng.normal(0,noise,h.sum())
    hist=[metrics(phi,ref[0],mask)]

    for t in range(len(ref)-1):
        if mode=="trajectory_clamp":
            phi[mask]=ref[t][mask]
            old[mask]=(target_old if t==0 else ref[t-1])[mask]
        new,new_old=step(phi,old,p)
        if mode=="trajectory_clamp":
            new[mask]=ref[t+1][mask]
        phi,old=new,new_old
        hist.append(metrics(phi,ref[t+1],mask))
    return phi,hist


def summarize(hist):
    body=hist[1:] if len(hist)>1 else hist
    out={}
    for k in body[0]:
        v=np.array([m[k] for m in body])
        out["mean_"+k]=float(v.mean())
        out["final_"+k]=float(v[-1])
    return out


def plot_curves(rows,outdir):
    modes=sorted({r["mode"] for r in rows})
    kinds=sorted({r["mask_kind"] for r in rows})
    fig,axs=plt.subplots(1,len(modes),figsize=(7*len(modes),5),squeeze=False)
    for ax,mode in zip(axs[0],modes):
        for kind in kinds:
            sub=[r for r in rows if r["mode"]==mode and r["mask_kind"]==kind]
            fs=sorted({r["fraction"] for r in sub})
            ys=[]; es=[]
            for f in fs:
                vals=[r["mean_hidden_corr"] for r in sub if r["fraction"]==f]
                ys.append(np.mean(vals)); es.append(np.std(vals))
            ax.errorbar(np.array(fs)*100,ys,yerr=es,marker="o",label=kind)
        ax.axhline(0,lw=.8)
        ax.set_xscale("log"); ax.set_ylim(-1.05,1.05)
        ax.set_xlabel("observed / locked field (%)")
        ax.set_ylabel("mean hidden-region correlation")
        ax.set_title(mode.replace("_"," "))
        ax.grid(alpha=.25); ax.legend()
    fig.tight_layout()
    fig.savefig(outdir/"completion_curves.png",dpi=180)
    plt.close(fig)


def plot_reference_masks(target,fractions,outdir):
    f=min(fractions,key=lambda x:abs(x-.10))
    kinds=["random","center","boundary","ring"]
    fig,axs=plt.subplots(1,5,figsize=(18,4))
    vmax=np.percentile(np.abs(target),99.5)
    axs[0].imshow(target,cmap="RdBu_r",vmin=-vmax,vmax=vmax); axs[0].set_title("reference")
    axs[0].axis("off")
    for i,k in enumerate(kinds,1):
        rng=np.random.default_rng(100+i)
        m=make_mask(k,target.shape[0],f,rng)
        axs[i].imshow(np.where(m,target,np.nan),cmap="RdBu_r",vmin=-vmax,vmax=vmax)
        axs[i].set_title(f"{k}: {100*f:g}%"); axs[i].axis("off")
    fig.tight_layout(); fig.savefig(outdir/"reference_and_masks.png",dpi=180); plt.close(fig)


def plot_examples(target,target_old,ref,p,fractions,outdir):
    f=min(fractions,key=lambda x:abs(x-.05))
    kinds=["random","boundary","center","ring"]
    fig,axs=plt.subplots(4,4,figsize=(14,13))
    vmax=np.percentile(np.abs(ref[-1]),99.5)
    for i,k in enumerate(kinds):
        rng=np.random.default_rng(1000+i)
        m=make_mask(k,target.shape[0],f,rng)
        cut=np.where(m,target,0)
        free,_=reconstruct(target,target_old,ref,m,p,"initial_only",0,rng)
        clamp,_=reconstruct(target,target_old,ref,m,p,"trajectory_clamp",0,rng)
        imgs=[cut,ref[-1],free,clamp]
        titles=[f"{k}: {100*f:g}% cue","reference future","initial-only","trajectory-clamped"]
        for j,(img,title) in enumerate(zip(imgs,titles)):
            axs[i,j].imshow(img,cmap="RdBu_r",vmin=-vmax,vmax=vmax)
            axs[i,j].set_title(title); axs[i,j].axis("off")
    fig.tight_layout(); fig.savefig(outdir/"example_reconstructions.png",dpi=180); plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--grid",type=int,default=96)
    ap.add_argument("--target-steps",type=int,default=450)
    ap.add_argument("--horizon",type=int,default=70)
    ap.add_argument("--fractions",type=float,nargs="+",default=[.50,.25,.10,.05,.02,.01])
    ap.add_argument("--mask-kinds",nargs="+",default=["random","center","boundary","ring"])
    ap.add_argument("--modes",nargs="+",default=["initial_only","trajectory_clamp"])
    ap.add_argument("--seeds",type=int,default=3)
    ap.add_argument("--hidden-noise",type=float,default=0.0)
    ap.add_argument("--outdir",type=Path,default=Path("phiworld_completion_results"))

    ap.add_argument("--dt",type=float,default=.08)
    ap.add_argument("--damping",type=float,default=.001)
    ap.add_argument("--base-c-sq",type=float,default=1.0)
    ap.add_argument("--tension-factor",type=float,default=5.0)
    ap.add_argument("--potential-lin",type=float,default=1.0)
    ap.add_argument("--potential-cub",type=float,default=.2)
    ap.add_argument("--biharmonic-gamma",type=float,default=.02)
    a=ap.parse_args()

    a.outdir.mkdir(parents=True,exist_ok=True)
    p=Params(a.dt,a.damping,a.base_c_sq,a.tension_factor,a.potential_lin,a.potential_cub,a.biharmonic_gamma)

    print("Building reference...")
    target,target_old,ref=build_reference(a.grid,p,a.target_steps,a.horizon)

    rows=[]
    total=a.seeds*len(a.mask_kinds)*len(a.fractions)*len(a.modes)
    runcount=0
    for seed in range(a.seeds):
        for kind in a.mask_kinds:
            for fraction in a.fractions:
                ms=10000*seed+101*a.mask_kinds.index(kind)+int(1e6*fraction)
                mask=make_mask(kind,a.grid,fraction,np.random.default_rng(ms))
                for mode in a.modes:
                    runcount+=1
                    final,hist=reconstruct(
                        target,target_old,ref,mask,p,mode,a.hidden_noise,
                        np.random.default_rng(ms+777*a.modes.index(mode))
                    )
                    agg=summarize(hist)
                    row={
                        "seed":seed,"mode":mode,"mask_kind":kind,
                        "fraction":float(mask.mean()),
                        "observed_cells":int(mask.sum()),
                        "hidden_cells":int((~mask).sum()),
                        **agg
                    }
                    rows.append(row)
                    print(f"[{runcount:3d}/{total}] {mode:16s} {kind:8s} {100*mask.mean():6.2f}% "
                          f"mean corr={agg['mean_hidden_corr']:+.4f} final={agg['final_hidden_corr']:+.4f}")

    with (a.outdir/"completion_results.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    summary={
        "params":asdict(p),"grid":a.grid,"target_steps":a.target_steps,"horizon":a.horizon,
        "seeds":a.seeds,
        "interpretation":{
            "initial_only":"partial state supplied once, then free evolution",
            "trajectory_clamp":"same spatial subset continuously synchronized to the reference",
            "score":"computed only on hidden cells",
        },
        "cue_thresholds":{}
    }
    for mode in a.modes:
        summary["cue_thresholds"][mode]={}
        for kind in a.mask_kinds:
            summary["cue_thresholds"][mode][kind]={}
            sub=[r for r in rows if r["mode"]==mode and r["mask_kind"]==kind]
            fs=sorted({r["fraction"] for r in sub})
            for th in [.25,.50,.75,.90]:
                passing=[]
                for f in fs:
                    vals=[r["mean_hidden_corr"] for r in sub if r["fraction"]==f]
                    if np.mean(vals)>=th: passing.append(f)
                summary["cue_thresholds"][mode][kind][str(th)]=min(passing) if passing else None

    with (a.outdir/"completion_summary.json").open("w",encoding="utf-8") as f:
        json.dump(summary,f,indent=2)

    plot_curves(rows,a.outdir)
    plot_reference_masks(target,a.fractions,a.outdir)
    plot_examples(target,target_old,ref,p,a.fractions,a.outdir)

    print("\nInterpretation:")
    print(" initial_only success  -> a fragment regenerates the future trajectory.")
    print(" trajectory_clamp success -> sparse ongoing observations synchronize the hidden field.")
    print(" random sparse > center -> more distributed constraint / more hologram-like.")
    print(" only one special patch works -> privileged seed, not distributed redundancy.")
    print(" nothing works -> also a valid result: PhiWorld2 may not be a completion attractor.")


if __name__=="__main__":
    main()
