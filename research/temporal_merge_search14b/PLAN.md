# Temporal checkpoint merge search, round 1

Objective: find 2–3 distinct method families that outperform TSG-TV. This is an
experimental objective, not a promised outcome. Different strengths of one
operator do not count as different methods. No change to the completed v2 study.

## Evidence and boundaries

All 57 dev tasks and their eight source / four merge results have already been
examined. They are development data. Additional seeds measure sampling stability
but do not make dev an untouched holdout. The historical TSG-TV Mean@8 is
78.5088%; a fresh concurrent TSG-TV control is required for this search.

The original 168-task test_normal split is reserved for confirmation of frozen
finalists. Do not inspect its model scores to choose recipes or retry until a
desired outcome appears. Before opening it, freeze finalists, baseline/control
models, seeds and the analysis. Report every selected method, including failures.
Test-normal task-family overlap with dev must be inventoried: an unseen task
split is not automatically an unseen-family split. No independent training-run
claim can be made from this single training trajectory.

## Three proposed operators

These are hypotheses. Names are local experiment labels, not claims of priority.
They use the existing source success matrix as calibration and produce one dense
static checkpoint. There is no inference-time router or additional LLM training.

1. **QIS: quality-guided increment surgery.** For each negative-quality interval,
   estimate its negative-effect probability with a paired task-family bootstrap.
   Form a beneficial direction from preceding positive-quality increments only.
   Remove the interval only at coordinates opposing that direction, with the
   smooth overlap 2*min(abs(d),abs(b))/(abs(d)+abs(b)). Keep aligned changes.
   Confidence is max(0,2*P(delta_quality<0)-1). This is an association-based
   editing heuristic, not a reconstruction of counterfactual training.
2. **DRR: detrended reward response.** Across all eight chronological checkpoints,
   remove intercept and linear time from quality and parameter displacements.
   Regress residual parameter changes on residual quality. Request the quality
   difference between the best source and latest source at fixed latest time.
   Shrink the response by its partial explained-variance ratio. This asks whether
   quality-associated variation beyond time is useful; it is not causal evidence.
3. **BCM: bootstrap consensus merge.** Fit a KL-regularized log-utility mixture of
   source success probabilities, then refit on family-bootstrap calibration
   samples. Use the mean weights as the target and their full covariance to
   attenuate coordinates with unstable anchor-relative updates:
   gate = mean_update^2/(mean_update^2 + beta*update_variance).
   Include the exact same mean-weight linear target as an operator control.
   This bootstrap does not measure inference-session uncertainty or certify a
   posterior over neural-network parameters.

QIS and DRR coordinate updates are capped at twice the absolute update of the
historical quality-weighted linear target. Initial strengths are 0.5 and 1.0.
BCM uncertainty penalties are 0.5 and 1.0. These six recipes, their calibration
matrix hashes and parameters are frozen before any candidate GPU evaluation.
The BCM linear target is a seventh export/control. All exports are FP32 with
unchanged architecture/tokenizer and complete ready manifests.

## Initial evaluation budget and decision rule

Eight new dev sessions: concurrent TSG-TV, BCM linear control, and six candidates.
Every cell uses 57 tasks x8, T=0.9, FP32/TRITON_ATTN, context 25580, prompt 20480,
response 4096, max 30 actions, TP=1 on the same eight GPUs, scheduler budget 8192.
All cells use seed 20261110, keeping settings symmetric. Equal seeds do not
guarantee identical random draws after generated histories diverge.

No candidate is eliminated on partial results. Compare full-task Mean@8 to the
concurrent TSG-TV and the historical 78.5088%; also report Pass@8, per-family
differences, update norms, runtime and storage cost. Use paired family bootstrap
intervals and label round-1 results exploratory and unadjusted. Rank strengths
within each family only after all eight cells finish. There is no entitlement to
declare three winners if only zero or one method improves.

After round 1, either freeze up to one recipe per promising family for independent
confirmation, or preregister a further bounded development round with a stated
reason. Confirmation needs a contemporaneous TSG-TV, best source and appropriate
linear controls; report simultaneous comparison uncertainty and all frozen seeds.
Distinguish positive point estimates, reproducible dev gains, and confirmed
holdout gains. If the evidence cannot support 2–3 winners, report that fact and
keep the objective open instead of manufacturing significance.

## Relevant prior work and novelty limits

- LAWA: https://arxiv.org/abs/2209.14981
- WSM: https://arxiv.org/abs/2507.17634
- QAM: https://arxiv.org/abs/2609.35168
- Temporal Sampling: https://arxiv.org/abs/2505.20196
- GTR-Turbo: https://arxiv.org/abs/2512.13043
- SimMerge: https://arxiv.org/abs/2601.09473
- AgentPatch: https://arxiv.org/abs/2608.06699
- Uncertainty-based gradient matching: https://arxiv.org/abs/2310.12808

Temporal averaging, uncertainty weighting, regression, sign-conflict reduction
and automatic source selection are established ideas. Publication claims require
deeper comparison and evidence for the specific temporal/behavioral mechanism;
numerically distinct formulas alone are not sufficient novelty.
