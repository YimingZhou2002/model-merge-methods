# Model Merge Methods Reproduction

Reproduction code and frozen AppWorld dev@8 artifacts for MMR, TSG-TV, and
BCM on the Qwen3-14B checkpoint trajectory. The code is kept under the same
`research/` module paths as the source experiments so the original launch and
audit flow remains recognizable.

The project owner/author is `yimingzhou2002`. See [`reproduce.md`](reproduce.md)
for setup, exact commands, result interpretation, and the difference between
reproducing the recorded result and running a fresh confirmation.

This repository contains code, small manifests, recipes, and aggregate results.
It does not contain model weights, AppWorld task data, or runtime environments.

## Included

- `research/checkpoint_merge_14b/`: source-bank selection, MMR-TIES and TSG-TV
  export, AppWorld evaluation controller, and independent audits.
- `research/temporal_merge_search14b/`: BCM calibration/export/evaluation and
  independent audits for its registered development rounds.
- `research/model_merge/weighted_merge.py`: streaming safetensors merge engine.
- `results/`: frozen recipes, manifests, result tables, and protocol amendments.
- `tests/`: numerical and export regression tests.

The implementation baseline is the local `agent_merge` main commit recorded in
[`provenance.json`](provenance.json). MMR/TSG and BCM study code was added from
the completed 14B experiment implementation; its source commit is recorded
there as well.
