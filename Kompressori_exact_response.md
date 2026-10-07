# Kompressori: exact response propagation and a local rank bound

**Research note — 7 October 2026**

## Result

The low effective rank reported by Kompressori Gate 6 survives replacing its finite-difference response probes with the analytic tangent equation. On the same 40 × 40 field, 100 overlapping Gaussian input directions, 24-step preparation delay, 25-step response horizon, and 30 outcome-independent event locations:

| Quantity | Committed finite-difference receipt | Fresh analytic-tangent calculation |
|---|---:|---:|
| Baseline directions for 95% spectral energy | 91 / 100 | 91 / 100 |
| Median update directions for 95% energy | 5 / 100 | 5 / 100 |
| Median update effective rank | 2.7531755 | 2.7682726 |
| Median update / baseline Frobenius norm | 0.1547755 | 0.1547298 |

This is a fresh numerical confirmation in the **same sampled input span**, not a theorem about the full response operator over every possible perturbation. “Exact tangent” means the derivative is propagated analytically rather than approximated by a small finite input difference; floating-point arithmetic remains.

An elementary consequence of the implemented equations is also proved below: changing one field cell changes at most five rows of the next-step state Jacobian. Consequently that **one-step Jacobian update has rank at most five**. The observed longer-horizon rank is a separate empirical fact: propagation can spread and accumulate these local changes.

The matrix identities used here are standard algebra. No novelty claim, biological claim, or connection to a Navier–Stokes singularity follows from this calculation.

## Source and frozen setup

The inspected repository is [anttiluode/Kompressori](https://github.com/anttiluode/Kompressori). The engine used here matches the core file at commit **7cca26e6677e2f6ee6351f320555102baf7bd8f9** (Git blob **5a8bdec8276279f23a4162309adaf4dfe2a08785**).

Relevant sources:

- [Field equations](https://github.com/anttiluode/Kompressori/blob/7cca26e6677e2f6ee6351f320555102baf7bd8f9/kompressori.py)
- [Gate 6 implementation](https://github.com/anttiluode/Kompressori/blob/7cca26e6677e2f6ee6351f320555102baf7bd8f9/gate6_lowrank_update.py)
- [Gate 6 receipt](https://github.com/anttiluode/Kompressori/blob/7cca26e6677e2f6ee6351f320555102baf7bd8f9/results/gate6/receipt.json)
- [Gate 7 receipt](https://github.com/anttiluode/Kompressori/blob/7cca26e6677e2f6ee6351f320555102baf7bd8f9/results/gate7/receipt.json)

The preparation event has amplitude 0.1 and is applied equally to the present and previous fields. Later response packets are likewise displacement packets, with equal perturbations of those two state components. The input Gram matrix is whitened exactly as in Gate 6. The floor used in whitening is retained.

The calculation ran with Python 3.12.14 and NumPy 2.3.5. It changed no GitHub files.

## 1. Analytic tangent equation

Let $L$ be the periodic five-point Laplacian, and write the acceleration as

$$
f(\phi)=c(\phi)\odot L\phi+\lambda\phi-\mu\phi^{\odot3}-\beta L^2\phi,
\qquad
c(u)=\frac{c_0^2}{1+\alpha u^2+10^{-9}}.
$$

Here $\lambda=1$, $\mu=0.2$, $\alpha=5$, $c_0^2=1$, and $\beta=0.02$. The time step is $h=0.08$ and damping is $\eta=0.001$.

Set

$$
d=1-\eta h,\qquad
q(\phi)=c'(\phi)\odot L\phi+\lambda\mathbf1-3\mu\phi^{\odot2}.
$$

Differentiating the code gives

$$
Df(\phi)v
=\mathrm{diag}(c(\phi))Lv
+\mathrm{diag}(q(\phi))v-\beta L^2v.
$$

For the state $s=(\phi,\phi_{\mathrm{old}})$, the next-step Jacobian is therefore

$$
K(\phi)=
\begin{pmatrix}
(1+d)I+h^2\bigl[\mathrm{diag}(c)L+
\mathrm{diag}(q)-\beta L^2\bigr] & -dI\\
I&0
\end{pmatrix}.
$$

The probe propagates perturbations with this equation while evolving the ordinary field with the unchanged engine. Responses are collected after each step, excluding the directly injected input at time zero, matching Gate 6.

## 2. Local row decomposition and rank bound

Compare two present fields $\phi_A$ and $\phi_0$. Terms independent of the present field cancel:

$$
\Delta K=h^2
\begin{pmatrix}
\mathrm{diag}(\Delta c)L+\mathrm{diag}(\Delta q)&0\\
0&0
\end{pmatrix}.
$$

Let $S$ contain the sites where $\Delta c_i$ or $\Delta q_i$ is nonzero. Every changed row is a rank-one contribution, so

$$
\mathrm{rank}(\Delta K)\le |S|.
$$

If the two fields differ at just one cell $j$, then:

1. $\Delta c$ is supported only at $j$.
2. At every other cell, $c'$ and the local potential derivative are unchanged.
3. $\Delta L\phi$ is supported at $j$ and its four nearest neighbours.
4. Hence $\Delta q$ is supported on those same five cells.

Thus, on the 40 × 40 periodic grid,

$$
\boxed{\mathrm{rank}(\Delta K)\le5.}
$$

The biharmonic term does not enlarge this bound: its Jacobian is state-independent and cancels in the difference. Changes to the previous field alone also do not change this next-step Jacobian, because the previous-field dependence is affine.

A generic seven-by-seven numerical example attained rank five. The explicit row decomposition agreed with the directly constructed Jacobian difference to relative error $3.69\times10^{-14}$.

**Boundary:** the repository's Gaussian preparation packets have nonzero tails throughout the grid. They are not literal one-cell writes. The rank-five theorem therefore does not directly certify their finite-time update. It identifies the local structure of the implemented sensitivity law.

## 3. What happens over several steps

Let $K_t^A$ and $K_t^0$ be the Jacobians along the prepared and unprepared trajectories. For a final-time state response, the exact telescoping identity is

$$
P_H^A-P_H^0
=\sum_{t=0}^{H-1}
\bigl(K_{H-1}^A\cdots K_{t+1}^A\bigr)
\bigl(K_t^A-K_t^0\bigr)
\bigl(K_{t-1}^0\cdots K_0^0\bigr).
$$

Empty products are identities.

Each term first transports an incoming perturbation to time $t$, passes it through a changed local sensitivity, and transports the result onward. Rank cannot increase when a term is multiplied on either side, but adding contributions can increase their joint rank. In particular,

$$
\mathrm{rank}(P_H^A-P_H^0)
\le \min\left(2N,\sum_{t=0}^{H-1}|S_t|\right).
$$

This explains why locality is a useful starting structure but does not guarantee a fixed tiny rank over a long horizon. A compact approximation also requires the transported contributions to remain concentrated in a small number of modes.

If each local difference is approximated by $\widehat{\Delta K_t}$, using the true surrounding propagators gives the error certificate

$$
\Vert \Delta P_H-\widehat{\Delta P_H}\Vert _2
\le \sum_t
\Vert P_{\mathrm{after},t}^A\Vert _2
\Vert \Delta K_t-\widehat{\Delta K_t}\Vert _2
\Vert P_{\mathrm{before},t}^0\Vert _2.
$$

This is an a posteriori bound, not automatically a cheap algorithm: the surrounding propagators must still be obtained or bounded.

A six-step, four-by-four dense-matrix check verified the telescoping identity to relative error $1.07\times10^{-13}$. For a whole response trajectory, apply the same identity to each observation time and stack the outputs.

## 4. Fresh numerical checks

The analytic trajectory derivative was compared with central differences of the unchanged engine over six steps. Relative errors were:

| Central-difference amplitude | Relative error |
|---:|---:|
| $10^{-3}$ | $6.79\times10^{-7}$ |
| $10^{-4}$ | $6.79\times10^{-9}$ |
| $10^{-5}$ | $7.72\times10^{-11}$ |

The quadratic convergence supports the derivative implementation.

For the frozen Gate-6 setup, the analytic baseline effective rank was **79.8289869**. Across all 30 fixed A locations, rank for 95% update energy ranged from **4 to 9**, with median **5**. The median effective rank was **2.7682726**, and median update/baseline norm was **0.1547298**.

Two additional pairs were chosen solely by the existing distance rule, not by their measured outcomes:

| Pair | Indices | Distance | Exact-tangent relative non-additivity |
|---|---|---:|---:|
| First closest pair | 0, 12 | 5.656854 | 1.085702 |
| Last farthest pair | 9, 29 | 28.284271 | $3.62\times10^{-13}$ |

These are two diagnostic checks, not a rerun of Gate 7's complete pair panel. A relative non-additivity above one is valid: the norm of the error can exceed the norm of the actual joint update.

## 5. A guard for the proposed overlap experiment

Gate 9 proposes asking whether low-rank subspace overlap predicts interference better than geometric distance. That is a valid empirical question, but subspace overlap alone cannot generally guarantee additivity.

For a simple counterexample, consider the smooth response family

$$
J(a,b)=
\begin{pmatrix}
1+a&\kappa ab\\
0&1+b
\end{pmatrix}.
$$

The isolated updates $aE_{11}$ and $bE_{22}$ have orthogonal input and output subspaces. Nevertheless the joint update contains the nonzero interaction $\kappa abE_{12}$.

For any smooth response family, the interaction is exactly

$$
J(a,b)-J(a,0)-J(0,b)+J(0,0)
=\int_0^a\int_0^b
\frac{\partial^2J}{\partial u \partial v}(u,v) dv du.
$$

That mixed curvature is the direct mathematical object behind non-additivity. Under bounded derivatives, the absolute interaction is $O(ab)$ near zero. This provides a clean amplitude-scaling test alongside an overlap predictor.

The counterexample does not disprove overlap as a useful predictor in Kompressori. It prevents promoting an empirical correlation into an unjustified general law.

## Most useful next question

Freeze a compressed update representation on development event locations, then test whether it predicts the response changes at unseen locations and for held-out input directions. Compare it with direct local-state predictors and simple translated-template controls. The present result covers a fixed 100-direction span; a predictor needs tests beyond that span before claiming general response preservation.

For interference, compare distance, input/output subspace overlap, and a predefined mixed-curvature estimate. Require held-out pairs and an amplitude-scaling control. None of these predictive gates has been run in this note.

## Reproduce

Clone Kompressori and check out commit 7cca26e6677e2f6ee6351f320555102baf7bd8f9. Install NumPy. Copy the Python appendix into exact_response_probe.py at the repository root and run:

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python exact_response_probe.py

The script prints derivative checks, the one-site bound check, the product identity check, all-30-location spectral summaries, and the two geometric-extreme pair checks. It uses only NumPy and the pinned kompressori.py module.

## Python appendix



```python
"""Reproducible analytic-tangent diagnostic for Kompressori; see the accompanying note."""
import json
import numpy as np
import kompressori as k

p = k.Params()
def lapb(x):
    return (np.roll(x,1,-2)+np.roll(x,-1,-2)
            +np.roll(x,1,-1)+np.roll(x,-1,-1)-4*x)

def coefficients(phi):
    den = 1+p.tension_factor*phi*phi+1e-9
    c = p.base_c_sq/den
    cp = -2*p.base_c_sq*p.tension_factor*phi/(den*den)
    q = cp*k.lap(phi)+p.potential_lin-3*p.potential_cub*phi*phi
    return c,q

def response(phi, old, packets, horizon=25):
    d = np.asarray(packets).copy()
    od = d.copy()
    history=[]
    phi,old=phi.copy(),old.copy()
    for _ in range(horizon):
        c,q=coefficients(phi)
        ld=lapb(d)
        nd=(2-p.damping*p.dt)*d-(1-p.damping*p.dt)*od
        nd+=p.dt**2*(c*ld+q*d-p.biharmonic_gamma*lapb(ld))
        phi,old=k.step(phi,old,p)
        d,od=nd,d
        history.append(d.copy())
    a=np.asarray(history)
    return a.transpose(0,2,3,1).reshape(-1,len(packets))

def advance(phi,old,steps):
    for _ in range(steps):
        phi,old=k.step(phi,old,p)
    return phi,old

def packet(n,x,y):
    yy,xx=np.mgrid[:n,:n]
    dx0=np.abs(xx-x);dy0=np.abs(yy-y)
    dx=np.minimum(dx0,n-dx0);dy=np.minimum(dy0,n-dy0)
    g=np.exp(-(dx*dx+dy*dy)/(2*(n/18)**2))
    return g/np.sqrt(np.mean(g*g))

def spectrum(cols,W):
    a=cols@W
    values=np.maximum(np.linalg.eigvalsh(a.T@a)[::-1],0)
    return dict(rank95=int(np.searchsorted(np.cumsum(values)/values.sum(),.95)+1),
                effective_rank=float(values.sum()**2/(values@values)),
                norm=float(np.sqrt(values.sum())))

# Check the analytic derivative against the unchanged engine.
rng=np.random.default_rng(8841)
phi=rng.normal(0,.3,(7,7));old=rng.normal(0,.3,(7,7))
direction=rng.normal(size=(7,7))
analytic=response(phi,old,[direction],horizon=6)[:,0]
errors=[]
for eps in [1e-3,1e-4,1e-5]:
    plus=k.rollout(phi+eps*direction,old+eps*direction,p,6)
    minus=k.rollout(phi-eps*direction,old-eps*direction,p,6)
    central=((plus-minus)/(2*eps))[1:].ravel()
    errors.append(float(np.linalg.norm(central-analytic)/np.linalg.norm(analytic)))
assert errors[-1]<1e-8,errors

# Directly check the one-site rank bound and product identity.
def lap_matrix(n):
    return lapb(np.eye(n*n).reshape(n*n,n,n)).reshape(n*n,n*n).T
def jacobian(phi,L):
    c,q=coefficients(phi);N=phi.size;I=np.eye(N)
    A=(2-p.damping*p.dt)*I
    A+=p.dt**2*(c.ravel()[:,None]*L+np.diag(q.ravel())-p.biharmonic_gamma*(L@L))
    return np.block([[A,-(1-p.damping*p.dt)*I],[I,np.zeros((N,N))]])

L=lap_matrix(7);prepared=phi.copy();prepared[3,3]+=.1
J0=jacobian(phi,L);JA=jacobian(prepared,L);delta=JA-J0
c0,q0=coefficients(phi);ca,qa=coefficients(prepared)
pred=np.zeros_like(delta)
pred[:49,:49]=p.dt**2*((ca-c0).ravel()[:,None]*L+np.diag((qa-q0).ravel()))
row_identity_error=float(np.linalg.norm(delta-pred)/np.linalg.norm(delta))
rank=int(np.linalg.matrix_rank(delta,tol=1e-12))
assert rank<=5 and row_identity_error<1e-12,(rank,row_identity_error)

nsmall=4;L4=lap_matrix(nsmall);H=6
ph=rng.normal(0,.3,(nsmall,nsmall));oh=ph.copy()
pa=ph.copy();pa[1,1]+=.1;oa=oh.copy()
KA=[];K0=[]
for _ in range(H):
    K0.append(jacobian(ph,L4));KA.append(jacobian(pa,L4))
    ph,oh=k.step(ph,oh,p);pa,oa=k.step(pa,oa,p)
I=np.eye(2*nsmall*nsmall)
prefix=[I]
for J in K0:prefix.append(J@prefix[-1])
suffix=[None]*(H+1);suffix[H]=I
for t in range(H-1,-1,-1):suffix[t]=suffix[t+1]@KA[t]
expanded=sum(suffix[t+1]@(KA[t]-K0[t])@prefix[t] for t in range(H))
product_error=float(np.linalg.norm((suffix[0]-prefix[-1])-expanded)/np.linalg.norm(suffix[0]-prefix[-1]))
assert product_error<1e-10,product_error
print(json.dumps(dict(derivative_central_relative_errors=errors,
                     one_site_jacobian_rank=rank,
                     row_identity_relative_error=row_identity_error,
                     telescope_relative_error=product_error)),flush=True)

# Reproduce G6 in the exact epsilon_B -> 0 limit, at all 30 fixed sites.
n=40
target,previous=k.evolve_state(n,p,350)
base_phi,base_old=advance(target.copy(),previous.copy(),24)
coords=[(x,y) for y in range(0,n,4) for x in range(0,n,4)]
packets=[packet(n,x,y) for x,y in coords]
inputs=np.stack([g.ravel() for g in packets],axis=1)
vals,vecs=np.linalg.eigh(inputs.T@inputs)
W=vecs@np.diag(1/np.sqrt(np.maximum(vals,max(vals.max()*1e-12,1e-15))))@vecs.T
base=response(base_phi,base_old,packets)
base_s=spectrum(base,W)
sites=[((7+13*i)%n,(3+17*i)%n) for i in range(30)]
rows=[]
cache={}
for i,(x,y) in enumerate(sites):
    g=packet(n,x,y)
    pp,po=advance(target+.1*g,previous+.1*g,24)
    delta=response(pp,po,packets)-base
    ds=spectrum(delta,W)
    rows.append(dict(i=i,x=x,y=y,**ds,norm_ratio=ds["norm"]/base_s["norm"]))
    if i in [0,8,20]:cache[i]=delta
print(json.dumps(dict(exact_base=base_s,
                     exact_update_rank95_median=float(np.median([r["rank95"] for r in rows])),
                     exact_update_effective_rank_median=float(np.median([r["effective_rank"] for r in rows])),
                     exact_update_norm_ratio_median=float(np.median([r["norm_ratio"] for r in rows])),
                     exact_update_rank95_range=[min(r["rank95"] for r in rows),max(r["rank95"] for r in rows)],
                     first_A=rows[0])),flush=True)

# One pair from each preexisting geometric extreme, selected without outcomes.
def distance(i,j):
    ax,ay=sites[i];bx,by=sites[j]
    dx=abs(ax-bx);dy=abs(ay-by)
    return float(np.hypot(min(dx,n-dx),min(dy,n-dy)))
pairs=sorted((distance(i,j),i,j) for i in range(30) for j in range(i+1,30))
pair_receipts=[]
for label,(dist,i,j) in [("first_near",pairs[0]),("last_far",pairs[-1])]:
    singles=[]
    for idx in [i,j]:
        if idx in cache:singles.append(cache[idx])
        else:
            g=packet(n,*sites[idx])
            pp,po=advance(target+.1*g,previous+.1*g,24)
            singles.append(response(pp,po,packets)-base)
    g=packet(n,*sites[i])+packet(n,*sites[j])
    pp,po=advance(target+.1*g,previous+.1*g,24)
    actual=(response(pp,po,packets)-base)@W
    summed=(singles[0]+singles[1])@W
    rel=float(np.linalg.norm(actual-summed)/np.linalg.norm(actual))
    pair_receipts.append(dict(pair=label,i=i,j=j,distance=dist,relative_nonadditivity=rel))
print(json.dumps(dict(exact_two_pair_check=pair_receipts)),flush=True)


```
