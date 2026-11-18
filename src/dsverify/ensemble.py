"""Ensemble verification tables built on dsverify.probabilistic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .probabilistic import (
    brier,
    crps_ensemble,
    exceedance_probability,
    rank_flatness,
    rank_histogram,
    spread_skill,
)


def member_array(members: pd.DataFrame, model: str):
    """(cases, members) wind-speed array and matching observations for one ensemble model."""
    sub = members[members["model"] == model]
    wide = sub.pivot_table(index=["station", "time"], columns="member", values="ws").sort_index()
    obs = sub.drop_duplicates(["station", "time"]).set_index(["station", "time"])["ws_obs"].reindex(wide.index)
    return wide.to_numpy(), obs.to_numpy(), wide.index


def ensemble_scores(matched: pd.DataFrame, members: pd.DataFrame, models, thresholds, reference=None):
    """Probabilistic scores for every model on the common sample.

    Ensemble models get CRPS, fair CRPS, spread-skill, rank-histogram flatness and Brier scores.
    Deterministic models are scored as one-member ensembles, so their CRPS equals the MAE and
    all models can be compared with one proper score.
    """
    ens_models = set(members["model"].unique())
    rows, rel_rows = [], []
    crps_by_model = {}
    for m in models:
        if m in ens_models:
            x, y, _ = member_array(members, m)
        else:
            sub = matched[matched["model"] == m]
            x, y = sub[["ws"]].to_numpy(), sub["ws_obs"].to_numpy()
        crps = crps_ensemble(y, x)
        crps_by_model[m] = float(np.nanmean(crps))
        row = {"model": m, "members": x.shape[1], "n": int(np.isfinite(crps).sum()), "crps": crps_by_model[m]}
        if x.shape[1] > 1:
            row["crps_fair"] = float(np.nanmean(crps_ensemble(y, x, fair=True)))
            row.update({k: v for k, v in spread_skill(y, x).items() if k not in ("n", "members")})
            fl = rank_flatness(rank_histogram(y, x))
            row.update({"rank_delta": fl["delta"], "outside_fraction": fl["outside_fraction"],
                        "expected_outside": fl["expected_outside"]})
        for thr in thresholds:
            b = brier(y, exceedance_probability(x, thr), thr)
            if b.get("n", 0) == 0:
                continue
            row[f"brier_{thr:g}"] = b["brier"]
            row[f"bss_{thr:g}"] = b["bss_clim"]
            for t in b["reliability_table"]:
                rel_rows.append({"model": m, "threshold": thr, **t})
        rows.append(row)
    prob = pd.DataFrame(rows)
    if reference in crps_by_model and crps_by_model[reference] > 0:
        prob["crpss_vs_ref"] = 1 - prob["crps"] / crps_by_model[reference]
    return prob, pd.DataFrame(rel_rows)
