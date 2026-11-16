"""Pure selection, trajectory gates, strict rollout validation and paired analysis."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


def select_bank(vectors, k=5, lam=0.7, tau=0.02):
    """Port of the frozen 8B MMR algorithm, with explicit input validation."""
    pool = sorted(vectors, key=int)
    if not 1 <= k <= len(pool) or not 0 <= lam <= 1 or not math.isfinite(tau) or tau <= 0:
        raise ValueError("Invalid MMR k/lambda/tau")
    vectors = {i: np.asarray(vectors[i], dtype=np.float64) for i in pool}
    if len({v.shape for v in vectors.values()}) != 1:
        raise ValueError("Task vectors must be aligned and have equal shape")
    if any(v.ndim != 1 or not v.size or not np.isfinite(v).all() or (v < 0).any() or (v > 1).any()
           for v in vectors.values()):
        raise ValueError("Invalid per-task success vector")
    quality = {i: float(vectors[i].mean()) for i in pool}
    similarity = {}
    for i in pool:
        similarity[i] = {}
        for j in pool:
            norm = np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j])
            similarity[i][j] = float(vectors[i] @ vectors[j] / norm) if norm else 0.0
    selected = [max(pool, key=quality.get)]
    trace = []
    while len(selected) < k:
        scores = {i: lam * quality[i] - (1-lam) * max(similarity[i][j] for j in selected)
                  for i in pool if i not in selected}
        best = max(scores, key=scores.get)
        trace.append({"selected_before": list(selected), "scores": scores, "chosen": best})
        selected.append(best)
    bank = sorted(selected, key=int)
    values = np.array([quality[i] for i in bank])
    weights = np.exp((values-values.max()) / tau)
    weights /= weights.sum()
    return {"quality": quality, "cosine": similarity, "greedy_order": selected,
            "trace": trace, "weights": dict(zip(bank, map(float, weights))),
            "same_bank_as_quality_only": set(bank) == set(sorted(pool, key=quality.get, reverse=True)[:k])}


def trajectory_gates(checkpoints, quality, update):
    """Exact TV/REV/QA coordinate formulas used in the 8B experiment."""
    if len(checkpoints) < 3 or len(checkpoints) != len(quality):
        raise ValueError("Need >=3 chronological checkpoints and aligned qualities")
    first = checkpoints[0]
    travel = torch.zeros_like(first)
    for left, right in zip(checkpoints, checkpoints[1:]):
        travel.add_((right-left).abs())
    tv = torch.where(travel > 0, 1-(checkpoints[-1]-first).abs()/travel.clamp_min(1e-30), 0.).clamp_(0, 1)
    early = checkpoints[-2]-first
    late = checkpoints[-1]-checkpoints[-2]
    opposed = ((early > 0) & (late < 0)) | ((early < 0) & (late > 0))
    rollback = ((update > 0) & (late < 0)) | ((update < 0) & (late > 0))
    reverse = 2*torch.minimum(early.abs(), late.abs())/(early.abs()+late.abs()).clamp_min(1e-30)
    reverse = reverse.clamp_(0, 1)*opposed*rollback
    q = torch.as_tensor(quality, dtype=torch.float64)
    q = q-q.mean()
    mean = torch.zeros_like(first, dtype=torch.float64)
    for model in checkpoints:
        mean.add_((model-first).double()/len(checkpoints))
    cov, var = torch.zeros_like(mean), torch.zeros_like(mean)
    for model, weight in zip(checkpoints, q):
        centered = (model-first).double()-mean
        cov.add_(centered*weight)
        var.add_(centered.square())
    denominator = (var*q.square().sum()).sqrt()
    correlation = torch.where(denominator > 0, cov/denominator.clamp_min(1e-30), 0.).clamp_(-1, 1).float()
    qa = tv*(1+update.sign()*correlation)/2
    return tv, reverse, qa.clamp_(0, 1)


def weighted_ties(base, donors, weights):
    """mergekit GTA/TIES: density=1, sum sign election, normalize=true, lambda=1."""
    weighted = torch.stack([(value-base)*weight for value, weight in zip(donors, weights)])
    elected = torch.where(weighted.sum(0) >= 0, 1., -1.)
    mask = weighted.sign() == elected
    w = torch.tensor(weights, dtype=base.dtype, device=base.device)
    w = w.reshape((-1,) + (1,)*base.ndim)
    divisor = (w*mask).sum(0)
    divisor = torch.where(divisor == 0, 1., divisor)
    return base+(weighted*mask).sum(0)/divisor


def read_rollouts(path: Path, *, tasks=57, samples=8, task_instructions=None):
    groups = defaultdict(list)
    prompts = {}
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Empty rollout log")
    all_ids = all(row.get("task_id") for row in rows)
    if any(row.get("task_id") for row in rows) and not all_ids:
        raise ValueError("Mixed task ID availability")
    for row in rows:
        if isinstance(row.get("score"), bool):
            raise ValueError("Boolean score is not an evaluation result")
        score = float(row["score"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid rollout score")
        if all_ids:
            key = str(row["task_id"])
        else:
            prompt = row.get("input")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("Missing initial task prompt")
            key = hashlib.sha256(prompt.encode()).hexdigest()
        groups[key].append(score)
        prompts[key] = row.get("input", "")
    if len(rows) != tasks*samples or len(groups) != tasks or any(len(g) != samples for g in groups.values()):
        raise ValueError(f"Incomplete task/sample matrix: rows={len(rows)}, groups={len(groups)}")
    keys = sorted(groups)
    result = {"task_keys": keys, "vector": [float(np.mean(groups[k])) for k in keys],
            "mean_at_8": float(np.mean([row["score"] for row in rows])),
            "pass_at_8": float(np.mean([max(groups[k]) > 0 for k in keys])),
            "rollouts": len(rows), "task_grouping": "task_id" if all_ids else "initial_input_sha256",
            "log_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    if task_instructions:
        families = []
        for key in keys:
            if all_ids and key in task_instructions:
                matched = {key.split("_")[0]}
            else:
                matched = {task_id.split("_")[0] for task_id, instruction in task_instructions.items()
                           if instruction and instruction in prompts[key]}
            if len(matched) != 1:
                break
            families.append(matched.pop())
        if len(families) == len(keys):
            result["task_families"] = families
    return result


def paired_comparison(left, right, *, iterations=20000, seed=20261109):
    if left["task_keys"] != right["task_keys"]:
        raise ValueError("Different task sets or prompts; refusing an unpaired comparison")
    delta = np.asarray(left["vector"])-np.asarray(right["vector"])
    rng = np.random.default_rng(seed)
    boot = delta[rng.integers(0, len(delta), size=(iterations, len(delta)))].mean(1)
    result = {"delta_pp": float(delta.mean()*100),
            "task_bootstrap_95_pp": (np.quantile(boot, [.025, .975])*100).tolist(),
            "improved_tasks": int((delta > 0).sum()), "worsened_tasks": int((delta < 0).sum()),
            "tied_tasks": int((delta == 0).sum()),
            "interpretation": "Exploratory reused dev, unadjusted task bootstrap, conditional on one session per model; not session replication or a task-family bootstrap."}
    families = left.get("task_families")
    if families is not None:
        if families != right.get("task_families") or len(families) != len(delta):
            raise ValueError("Task family alignment differs")
        indices = [np.flatnonzero(np.array(families) == family) for family in sorted(set(families))]
        family_sums = np.array([delta[i].sum() for i in indices])
        family_sizes = np.array([len(i) for i in indices])
        chosen = rng.integers(0, len(indices), size=(iterations, len(indices)))
        family_boot = family_sums[chosen].sum(1)/family_sizes[chosen].sum(1)
        result["family_bootstrap_95_pp"] = (np.quantile(family_boot, [.025, .975])*100).tolist()
        result["families"] = len(indices)
        result["interpretation"] = "Exploratory reused dev; unadjusted family-cluster and task bootstrap intervals conditional on one session per model, not session replication."
    return result
