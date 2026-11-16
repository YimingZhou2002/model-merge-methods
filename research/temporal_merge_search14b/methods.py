"""Calibration and bounded temporal operators; no imports from rollout code."""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize


def fit_portfolio(probabilities, prior, penalty=0.02):
    p, prior = np.asarray(probabilities, float), np.asarray(prior, float)
    if p.ndim != 2 or p.shape[1] != len(prior) or not np.isfinite(p).all():
        raise ValueError("Invalid task x source probability matrix")
    if (p <= 0).any() or (p > 1).any() or (prior <= 0).any() or not np.isclose(prior.sum(), 1):
        raise ValueError("Probabilities/prior must have positive support")

    def objective(w):
        safe = np.maximum(w, 1e-12)
        mixture = p @ w
        entropy = np.log(safe / prior)
        return (-np.log(mixture).mean() + penalty * (w @ entropy),
                -(p / mixture[:, None]).mean(0) + penalty * (entropy + 1))

    result = minimize(objective, prior.copy(), jac=True, method="SLSQP",
                      bounds=[(1e-10, 1)] * len(prior),
                      constraints={"type": "eq", "fun": lambda w: w.sum()-1,
                                   "jac": lambda w: np.ones_like(w)},
                      options={"ftol": 1e-11, "maxiter": 500})
    if not result.success:
        raise RuntimeError(f"Portfolio optimization failed: {result.message}")
    weights = np.maximum(result.x, 0)
    return weights / weights.sum()


def calibrate(vectors, families, *, seed=20261110, draws=256):
    """Source x task vectors; paired family draws retain all tasks in a family."""
    v = np.asarray(vectors, dtype=np.float64)
    if v.ndim != 2 or v.shape[0] < 3 or v.shape[1] != len(families):
        raise ValueError("Need aligned source x task vectors and families")
    if not np.isfinite(v).all() or (v < 0).any() or (v > 1).any() or draws < 2:
        raise ValueError("Invalid calibration scores/draw count")
    groups = [np.flatnonzero(np.asarray(families) == f) for f in sorted(set(families))]
    rng = np.random.default_rng(seed)
    boot = [np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
            for _ in range(draws)]
    quality = v.mean(1)
    prior = np.exp((quality-quality.max()) / .02)
    prior /= prior.sum()
    # Jeffreys smoothing of eight Bernoulli outcomes; retain task differences.
    p = (v.T * 8 + .5) / 9
    weights = np.array([fit_portfolio(p[ix], prior) for ix in boot])
    mean = weights.mean(0)
    covariance = np.cov(weights, rowvar=False, ddof=1)
    eig, basis = np.linalg.eigh(covariance)
    factor = (basis * np.sqrt(np.maximum(eig, 0))).T
    dq = np.diff(quality)
    confidence = [max(0., 2*np.mean([np.diff(v[:, ix].mean(1))[j] < 0 for ix in boot])-1)
                  if dq[j] < 0 else 0. for j in range(len(dq))]
    time = np.linspace(-1, 1, len(quality))
    design = np.column_stack([np.ones(len(time)), time])
    projection = np.eye(len(time)) - design @ np.linalg.pinv(design)
    qres = projection @ quality
    qvar = float(qres @ qres)
    if qvar <= 1e-15:
        raise ValueError("Quality has no variation beyond linear time")
    return {"quality": quality.tolist(), "quality_differences": dq.tolist(),
            "negative_interval_confidence": confidence,
            "portfolio_weights": mean.tolist(), "portfolio_covariance": covariance.tolist(),
            "covariance_factor": factor.tolist(), "point_portfolio_weights": fit_portfolio(p, prior).tolist(),
            "residual_projection": projection.tolist(), "quality_residual": qres.tolist(),
            "quality_residual_norm2": qvar, "quality_target_gain": float(quality.max()-quality[-1]),
            "seed": seed, "bootstrap_draws": draws, "family_count": len(groups)}


def transform(values, calibration, historical_weights):
    """All outputs use FP32 centered displacements; flat coordinates stay exact."""
    if len(values) != len(calibration["quality"]) or len(values) != len(historical_weights):
        raise ValueError("Source count differs from calibration")
    if any(x.dtype != torch.float32 or x.shape != values[0].shape for x in values):
        raise ValueError("Expected equally shaped FP32 source tensors")
    anchor = values[-1]
    x = [v-anchor for v in values]
    weighted = lambda coefficients: sum((v*float(w) for v, w in zip(x, coefficients)), torch.zeros_like(anchor))
    reference_update = weighted(historical_weights)
    cap = 2*reference_update.abs()
    increments = [right-left for left, right in zip(values, values[1:])]
    dq = calibration["quality_differences"]
    rollback = torch.zeros_like(anchor)
    beneficial = torch.zeros_like(anchor)
    positive_mass = 0.
    for j, delta in enumerate(increments):
        confidence = calibration["negative_interval_confidence"][j]
        if confidence > 0 and positive_mass > 0:
            b = beneficial / positive_mass
            overlap = (2*torch.minimum(delta.abs(), b.abs()) /
                       (delta.abs()+b.abs()).clamp_min(1e-30))
            conflict = (delta*b < 0).to(delta.dtype)
            rollback.add_(confidence * overlap * conflict * delta)
        if dq[j] > 0:
            beneficial.add_(delta, alpha=float(dq[j]))
            positive_mass += dq[j]
    qis = torch.maximum(torch.minimum(-rollback, cap), -cap)

    qres = calibration["quality_residual"]
    qvar = calibration["quality_residual_norm2"]
    slope = weighted(qres) / qvar
    projection = calibration["residual_projection"]
    total_residual2 = torch.zeros_like(anchor)
    # Full residual variance after removing intercept and time; use all covariances.
    for row in projection:
        total_residual2.add_(weighted(row).square())
    explained = slope.square() * qvar
    reliability = torch.where(total_residual2 > 0,
                              explained / total_residual2.clamp_min(1e-30), 0.).clamp_(0, 1)
    response = slope * calibration["quality_target_gain"]
    drr = reliability * torch.maximum(torch.minimum(response, cap), -cap)

    update = weighted(calibration["portfolio_weights"])
    variance = torch.zeros_like(anchor)
    for row in calibration["covariance_factor"]:
        variance.add_(weighted(row).square())
    result = {"bcm_linear": anchor+update}
    for strength, suffix in [(.5, "05"), (1., "10")]:
        result[f"qis_{suffix}"] = anchor+strength*qis
        result[f"drr_{suffix}"] = anchor+strength*drr
        gate = torch.where(update.square()+strength*variance > 0,
                           update.square()/(update.square()+strength*variance).clamp_min(1e-30), 0.)
        result[f"bcm_{suffix}"] = anchor+gate*update
    return result
