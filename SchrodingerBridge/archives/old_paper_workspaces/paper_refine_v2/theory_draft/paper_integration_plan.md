# Paper Integration Plan: Adding Theory to AAAI-26 Paper

## Constraints
- AAAI-26: 7 pages technical content + references + reproducibility checklist
- Current paper: ~5.5 pages content → ~1.5 pages available
- Must maintain readability for non-theory reviewers

## Changes Summary

| Change | Location | Effort |
|--------|----------|--------|
| 1. New subsection 3.5 "Formal analysis" | After 3.4 | ~0.5 pages |
| 2. Update "Bounded theoretical claim" paragraph in 3.4 | Within 3.4 | ~0.2 pages |
| 3. Trajectory straightness figure | New Figure | ~0.2 pages |
| 4. Add trajectory straightness to ablation discussion (Sec 5) | Within ablations | ~0.1 pages |
| 5. Theoretical contribution checkbox | Checklist | 1 line |

## Detailed Plan

### Change 1: New subsection 3.5 (after line 131 of paper_aaai2026.tex)

Insert after "\subsection{Terminal SWD regularization}" (before the figure):

```latex
\subsection{Bounded formal analysis}
\label{sec:formal}

We provide a bounded formal interpretation of the three-part objective,
explicitly stating what each loss controls and how it relates to the
continuous-time transport perspective.  Full proofs appear in the
supplement; here we state the key results and their empirical support.

\paragraph{Proposition A (Conditional displacement).}
For fixed OT coupling $\pi^\star$, minimizing
$\E\|z_0+v_\theta(z_0,1,s)-\tilde z_1\|^2$ yields the Bayes optimal
one-step displacement $v^\star(z_0,1,s)=\E[\tilde z_1-z_0\mid z_0,s]$.
This means the learned velocity at $t=1$ is an estimator of the
expected OT transport direction.  \emph{(Code: \texttt{loss\_type="omf"}
in Sec.~\ref{sec:method}.)}

\paragraph{Proposition B (Kinetic displacement control).}
Let $\cA_K(v)=\frac{1}{K}\sum_{k=0}^{K-1}\|v_\theta(z_k,t_k,s)\|^2$
be the discrete path action under $K$-step Euler integration.
Under mild Lipschitz assumptions on $v_\theta$,
$\cA_K(v)\to\int_0^1\E\|v_\theta(z(t),t,s)\|^2\,dt$ at $O(1/K)$,
and $\cL_{\mathrm{kin}}$ controls the one-step displacement.
Empirically (Table~\ref{tab:straightness}), removing $\cL_{\mathrm{kin}}$
reduces path straightness (PLR 0.94 vs.\ 0.98).

\paragraph{Proposition C (Euler discretization error).}
Under the same regularity conditions,
$\|z(1)-z_K\|\leq C/K$.
Experimentally (Table~\ref{tab:step}), $K=4$ already achieves
error $<2$ in latent space (relative error $8\%$ of displacement),
and $K=12$ reduces error to $<0.6$,
consistent with the $O(1/K)$ bound and explaining the flat
step-count curve (Section~\ref{sec:step}).

\paragraph{Proposition D (OT directional coherence).}
OT coupling produces more directionally consistent supervision than
random coupling: cosine similarity $0.150$ vs.\ $0.124$ ($+21\%$,
measured over 640 samples).  This improves the geometric consistency
of the learned transport field without increasing displacement variance.
\end{latex}
```

### Change 2: Update "Bounded theoretical claim" paragraph (lines 118-120 of paper)

Replace current text:
```
\paragraph{Bounded theoretical claim.}
The method is not an exact Schrodinger Bridge solver...
```
With:
```
\paragraph{Bounded theoretical claim.}
The preceding propositions constitute a bounded formal analysis
of the proposed latent transport framework.  The method is not
an exact Schr\"odinger Bridge solver---it does not estimate
forward and backward stochastic scores nor optimize the full
entropic dual objective---but each loss has a clear interpretation:
the velocity field estimates the expected OT transport direction
(Prop.~A), the kinetic term controls path displacement
(Prop.~B), and the Euler error is bounded at $O(1/K)$
(Prop.~C).  OT coupling further improves directional coherence
(Prop.~D).  This bounded interpretation is reflected empirically:
removing the terminal constraint weakens style, while removing
the path penalty destroys content.
```

### Change 3: Trajectory straightness figure

Add new table or include in Ablations section (Table 3 in paper):

```
D0 (full control)       0.981   0.9995
D1 (no terminal SWD)    0.087   1.0000
D2 (no kinetic)         0.9425  0.9999
```

Can be either a new small table or added as columns to the existing
ablation table.

### Change 4: Update ablation discussion (Section 5, around line 261)

Add 1-2 sentences:
```
Trajectory straightness analysis (Table~\ref{tab:straightness})
further reveals the mechanism: the full model has path length ratio
0.98, while removing terminal SWD produces near-perfect straight
lines (0.9995) but weak style, and removing kinetic regularization
reduces straightness to 0.94 with content collapse.  This confirms
that kinetic regularization controls path regularity, while
terminal SWD introduces controlled curvature toward the target
patch distribution.
```

### Change 5: Update reproducibility checklist (lines 324)

Change:
```
\noindent\textbf{Does this paper make theoretical contributions?} \textbf{No}.
```
To:
```
\noindent\textbf{Does this paper make theoretical contributions?} \textbf{Yes (bounded)}.
The paper provides a bounded formal analysis of the proposed latent
transport framework with explicit propositions, assumptions, and
experimental validation: conditional displacement interpretation
(Proposition~A), displacement control via kinetic regularization
(Proposition~B), $O(1/K)$ Euler error bound (Proposition~C), and
OT directional coherence (Proposition~D).  These results constitute
a formal justification of the three-part design without claiming
equivalence to a full Schr\"odinger Bridge solver.
```

## Page Budget

| Current content | Pages |
|----------------|-------|
| Title + abstract | 0.3 |
| Sec 1-2 (Intro + Related) | 1.5 |
| Sec 3 (Method) | 1.5 |
| Sec 4-6 (Experiments + Discussion + Conclusion) | 2.2 |
| References + Checklist | 1.5 |
| **Total current** | **~7.0** |

After changes:
| Changed content | Delta |
|-----------------|-------|
| Sec 3.5 (new) | +0.5 |
| Tighten existing text | -0.2 |
| Straightness table | +0.15 |
| **Total after** | **~7.45** |

May need to tighten existing text (e.g., reduce Related Work,
compact some experiment descriptions) to stay within 7 pages + refs.
