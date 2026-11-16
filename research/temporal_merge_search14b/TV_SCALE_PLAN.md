# Preregistered TSG-TV scale controls and confirmation amendment

This plan is recorded after the concurrent TSG-TV reference completed and while
BCM-linear is still running, before any new candidate or linear-control score
has completed. It addresses the research objective's matched search-budget
requirement. Neither existing development round is changed or shortened.

## Bounded additional controls

Each proposed family has three registered settings: two original strengths and
one equal-norm/relative follow-up. Add two TSG-TV settings to the existing default:

- tsg_tv_half: anchor300 + 0.5 * (stored_TSG_TV - anchor300).
- tsg_tv_double: anchor300 + 2.0 * (stored_TSG_TV - anchor300).
- Default tsg_tv: the already exported model, scale 1.0.

This gives three distinct settings for TSG-TV as well. The default already has
two development seeds; its competitors have three settings evaluated once each.
Thus setting counts match, while TSG-TV has one extra default-reference session.
Report both counts explicitly; do not claim identical total GPU search costs.
The two new settings add exactly 912 development trajectories. No extra scale
or reward-conditioned range extension is allowed in this control round.

Build FP32 exports from the existing stored TV model and anchor. Independently
check complete output-shard hashes, float64 formula probes across every tensor,
all stored coordinates for finite values and actual update norms. Target update
norms are respectively 0.5 and 2 times the stored default TV norm; allow 0.2%
relative norm error for FP32 storage. Export and audit may overlap the original
GPU queues. No new GPU control may start until both original rounds finish and
their independent raw audits pass.

Use all 57 dev tasks x8, seed **20261112**, FP32, T=0.9, eight GPUs, TP1,
TRITON_ATTN, context25580/prompt20480/response4096, max30 actions,
scheduler4096 and memory0.80. Reuse the completed second-round default TV cell
as the common-seed reference. Preserve all data/task/experience settings and
check identical initial prompt keys. The new sessions run sequentially after
round two; temporal execution order differs and is disclosed.

Select the optimized TV setting by the largest integer success count among
these three same-seed cells. Exact ties use lexicographic recipe name, which
prefers the default when it ties. Preserve every result and attempt. This is a
bounded development choice, not proof of a globally optimal TV scaling.

## Explicit amendment to the first confirmation rules

The original `CONFIRMATION_RULES.md` and its registration remain preserved.
This preregistered amendment adds a search-matched TV comparator before any
new candidate/control score is available and before any 14B held-out outcome
is opened. It supersedes the following parts of those rules:

1. Require both original rounds **and this bounded TV-scale round** to finish
   and pass independent audits before freezing confirmation.
2. Choose the best member within each proposed family by its original
   own-round default-TV success delta, as originally specified. For development
   eligibility, additionally require that delta to exceed the optimized TV's
   same-seed gain over the default. This subtraction of development gains across
   rounds is a noisy selection heuristic, not a matched-seed inferential claim.
   At least two families must remain eligible before opening confirmation.
3. Include the optimized TV setting as an additional held-out comparator. If
   the default wins the TV search, alias this reference to the default's existing
   sessions instead of sampling an identical model again. Otherwise evaluate the
   additional model at both fixed seeds. Every included model retains all 168
   test-normal tasks x8 and both seeds **20261113 and 20261114**.
4. Use **12 planned comparisons** (up to three families x four references) for
   the nominal Bonferroni family-percentile intervals at alpha=0.05, with 20,000
   draws and analysis RNG seed **20261115**. Retain this conservative count even
   with fewer families or an aliased optimized-TV comparator.
5. A search-budget-matched confirmed improvement over TSG-TV requires positive
   adjusted lower bounds and positive per-seed Mean@8 deltas against **both the
   default and optimized TV references**. Report default-only improvements
   separately. The stronger all-control label also requires these conditions
   against optimized linear and checkpoint280.

At most seven distinct models enter confirmation: three families and four
controls. Its maximum budget becomes fourteen sessions and 18,816 trajectories;
it is smaller when fewer families qualify or the optimized TV aliases default.
This changes the prospective budget, not any completed score or inference setting.
All other boundaries, failure rules and interpretation limits remain in force.
If fewer than two families qualify, continue bounded development while preserving
the unopened 14B confirmation outcomes; the user goal remains incomplete.
