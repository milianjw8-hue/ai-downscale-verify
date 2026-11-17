"""Verification metrics for wind speed and direction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .wind import angular_difference, speed_dir_to_uv

QUANTILES = (0.5, 0.9, 0.95, 0.99)


def continuous_scores(obs, mod, wd_obs=None, wd_mod=None, calm_ms: float = 1.0) -> dict:
    """Bias, MAE, RMSE, correlation, variability and tail statistics of wind speed.

    Direction scores use only pairs where both speeds exceed `calm_ms`.
    """
    obs = np.asarray(obs, dtype=float)
    mod = np.asarray(mod, dtype=float)
    ok = np.isfinite(obs) & np.isfinite(mod)
    o, m = obs[ok], mod[ok]
    n = int(ok.sum())
    out = {"n": n}
    if n < 2:
        return out
    err = m - o
    out.update(
        mean_obs=o.mean(),
        mean_mod=m.mean(),
        bias=err.mean(),
        mae=np.abs(err).mean(),
        rmse=np.sqrt((err**2).mean()),
        corr=np.corrcoef(o, m)[0, 1] if o.std() > 0 and m.std() > 0 else np.nan,
        std_ratio=m.std() / o.std() if o.std() > 0 else np.nan,
        max_obs=o.max(),
        max_mod=m.max(),
    )
    for q in QUANTILES:
        tag = f"p{int(q * 100)}"
        out[f"{tag}_obs"] = np.quantile(o, q)
        out[f"{tag}_mod"] = np.quantile(m, q)
        out[f"{tag}_bias"] = out[f"{tag}_mod"] - out[f"{tag}_obs"]

    if wd_obs is not None and wd_mod is not None:
        wo = np.asarray(wd_obs, dtype=float)[ok]
        wm = np.asarray(wd_mod, dtype=float)[ok]
        good = np.isfinite(wo) & np.isfinite(wm) & (o > calm_ms) & (m > calm_ms)
        if good.sum() >= 2:
            d = angular_difference(wm[good], wo[good])
            out["dir_bias"] = _circular_mean(d)
            out["dir_mae"] = np.abs(d).mean()
            uo, vo = speed_dir_to_uv(o[good], wo[good])
            um, vm = speed_dir_to_uv(m[good], wm[good])
            out["vector_rmse"] = np.sqrt(((um - uo) ** 2 + (vm - vo) ** 2).mean())
            out["n_dir"] = int(good.sum())
    return out


def _circular_mean(deg):
    r = np.radians(deg)
    return np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean()))


def categorical_scores(obs, mod, threshold: float) -> dict:
    """2x2 contingency scores for the event `speed >= threshold`."""
    obs = np.asarray(obs, dtype=float)
    mod = np.asarray(mod, dtype=float)
    ok = np.isfinite(obs) & np.isfinite(mod)
    eo, em = obs[ok] >= threshold, mod[ok] >= threshold
    a = int((eo & em).sum())  # hits
    b = int((~eo & em).sum())  # false alarms
    c = int((eo & ~em).sum())  # misses
    d = int((~eo & ~em).sum())  # correct negatives
    n = a + b + c + d
    a_r = (a + b) * (a + c) / n if n else np.nan
    return {
        "threshold": threshold,
        "hits": a,
        "false_alarms": b,
        "misses": c,
        "correct_negatives": d,
        "pod": a / (a + c) if (a + c) else np.nan,
        "far": b / (a + b) if (a + b) else np.nan,
        "csi": a / (a + b + c) if (a + b + c) else np.nan,
        "fbi": (a + b) / (a + c) if (a + c) else np.nan,
        "ets": (a - a_r) / (a + b + c - a_r) if n and (a + b + c - a_r) else np.nan,
    }


def skill_vs_reference(obs, mod, ref, n_boot: int = 1000, block: int = 24, seed: int = 0) -> dict:
    """MSE skill score of `mod` relative to `ref` with a moving-block bootstrap CI.

    MSESS = 1 - MSE(mod) / MSE(ref). Positive means `mod` beats the reference.
    `block` is the block length in samples (24 for hourly data) to respect autocorrelation.
    Inputs must be aligned 1-D arrays from a single time series; for several stations
    pass station-concatenated arrays and blocks will mix stations only at boundaries.
    """
    obs, mod, ref = (np.asarray(x, dtype=float) for x in (obs, mod, ref))
    ok = np.isfinite(obs) & np.isfinite(mod) & np.isfinite(ref)
    o, m, r = obs[ok], mod[ok], ref[ok]
    n = o.size
    if n < 2:
        return {"n": n}
    se_m, se_r = (m - o) ** 2, (r - o) ** 2
    msess = 1 - se_m.mean() / se_r.mean() if se_r.mean() > 0 else np.nan
    out = {"n": n, "mse_mod": se_m.mean(), "mse_ref": se_r.mean(), "msess": msess}
    if n_boot and n >= 2 * block:
        rng = np.random.default_rng(seed)
        n_blocks = int(np.ceil(n / block))
        starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
        idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
        bm, br = se_m[idx].mean(1), se_r[idx].mean(1)
        boot = 1 - bm / br
        out["msess_lo95"], out["msess_hi95"] = np.quantile(boot, [0.025, 0.975])
        out["p_not_better"] = float((boot <= 0).mean())
    return out


def scores_table(matched: pd.DataFrame, models: list[str], thresholds, by_station: bool):
    """Build continuous and categorical score tables from the matched long table."""
    cont, cat = [], []
    groups = matched.groupby("station") if by_station else [("ALL", matched)]
    for st, g in groups:
        for name in models:
            sub = g[g["model"] == name]
            row = {"station": st, "model": name}
            row.update(continuous_scores(sub["ws_obs"], sub["ws"], sub["wd_obs"], sub["wd"]))
            cont.append(row)
            for thr in thresholds:
                crow = {"station": st, "model": name}
                crow.update(categorical_scores(sub["ws_obs"], sub["ws"], thr))
                cat.append(crow)
    return pd.DataFrame(cont), pd.DataFrame(cat)
