# Prospective equal-norm development round

This follow-up is specified before any new complete dev score is available.
It does not replace or eliminate any of the eight frozen round-1 sessions.
Launch only after that complete round and its raw audit have been examined.
All results remain development evidence; test-normal scores remain excluded.

## Motivation from parameter diagnostics, not reward selection

Round-1 actual stored update norms relative to checkpoint300 are approximately:
QIS-1.0 0.06611, DRR-1.0 0.04497, BCM-1.0 0.25739 and BCM-linear 0.26290.
Historical TSG-TV has norm 0.14034. Thus QIS/DRR start at substantially smaller
update energies, while the initial BCM penalties barely change its target norm.
These differences can confound an operator comparison. No new model score is
used to choose the normalized recipes below.

The numerical inputs are the completed round-1 export-statistics and independent
geometry/formula audits. Freeze their hashes with round-2 recipes. Keep all
round-1 outcomes, including poor strengths, in the eventual report.

## Fixed candidate formulas

Let a=checkpoint300, r=the actual stored TSG-TV update norm, and u_Q/u_D the
actual stored QIS-1.0/DRR-1.0 updates. Define:

- QIS-NM: a + (r/||u_Q||)*u_Q.
- DRR-NM: a + (r/||u_D||)*u_D.
- BCM-relative-NM: use the same frozen round-1 portfolio mean weights w and
  covariance C. For source updates x_i=theta_i-a, let u=sum_i w_i*x_i and
  v_coordinate=x_coordinate^T C x_coordinate. Define
  beta=||stored_BCM_linear-a||^2 / trace(C*G), where G_ij=<x_i,x_j> is the
  full source-update Gram matrix. This gives a relative uncertainty gate
  g=u^2/(u^2+beta*v), with g=0 when both terms vanish. Form the actual FP32
  raw checkpoint a+g*u, measure its stored update norm, then globally rescale
  its stored update to r. The raw checkpoint is retained for diagnostics, not
  selected by an extra GPU score.

The ratio beta sets the total mean-update energy equal to the scaled bootstrap
uncertainty energy before coordinate shrinkage. It is fixed by geometry, not a
reward-tuned penalty. Generic uncertainty/Wiener shrinkage and norm matching are
established ideas; novelty would need to come from the particular behavioral
source-weight uncertainty estimate and evidence for its temporal mechanism.

This round evaluates the same three families, not three additional inventions.
Reject a zero/near-zero raw update instead of dividing by it or claiming a new
method. Independently verify every export and actual norm matching within 0.2%.

## Controls and budget

Seven complete 57-task dev@8 sessions, all using seed 20261112, FP32, T=0.9,
context25580/prompt20480/response4096, max30 actions, eight GPUs, TP1,
TRITON_ATTN, memory0.80 and scheduler4096:

1. Concurrent TSG-TV reference.
2. Checkpoint280, the previously best observed source by Mean@8.
3. The original TSG-TV-norm global interpolation toward its historical linear
   target, retaining the already exported model.
4. BCM-linear-NM, global interpolation toward the exact stored round-1 BCM
   linear target at norm r.
5. QIS-NM.
6. DRR-NM.
7. BCM-relative-NM.

This is a bounded second development round: 3192 additional trajectories and no
extra score-based penalty sweep. Do not compare different seeds as if they were
matched sessions. Report all seven cells, primary Mean@8, secondary Pass@8,
paired task-family uncertainty, norm/geometry diagnostics, actual runtime and
all initial failures. Norm alignment controls total update size, not functional
KL or causal knowledge content; those claims need separate evidence.

Select at most one frozen finalist per family using development evidence. A
family is not a confirmed winner until it has appropriate independent validation
against TSG-TV and relevant simple controls, with the final multiplicity/seed
analysis frozen before examining new test-normal outcomes.
