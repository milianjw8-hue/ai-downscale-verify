"""End-to-end verification run driven by a YAML config."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import __version__, plots
from .grid import extract_at_stations, open_model
from .ensemble import ensemble_scores
from .metrics import scores_table, skill_vs_reference
from .wind import uv_to_speed_dir
from .obs import load_observations


def load_config(path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = Path(path).resolve().parent
    root = Path(cfg.get("root", base))
    if not root.is_absolute():
        root = (base / root).resolve()

    def _abs(p):
        if isinstance(p, list):
            return [_abs(x) for x in p]
        pp = Path(p)
        return str(pp if pp.is_absolute() else root / pp)

    cfg["obs"]["path"] = _abs(cfg["obs"]["path"])
    cfg["obs"]["station_meta"] = _abs(cfg["obs"]["station_meta"])
    for m in cfg["models"]:
        m["path"] = _abs(m["path"])
    cfg["output"] = _abs(cfg.get("output", "outputs"))
    return cfg


def _extract_model(mcfg: dict, stations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Extract one model at the stations. For an ensemble (`member_dim` in the config, or an
    `ensemble` dimension) returns the ensemble-mean table and the per-member table."""
    ds = open_model(mcfg)
    mdim = mcfg.get("member_dim", "ensemble")
    if mdim not in ds.dims:
        ext = extract_at_stations(ds, stations, mcfg)
        ext["model"] = mcfg["name"]
        return ext, None
    parts = []
    for k in range(ds.sizes[mdim]):
        e = extract_at_stations(ds.isel({mdim: k}), stations, mcfg)
        e["member"] = k
        parts.append(e)
    members = pd.concat(parts, ignore_index=True)
    members["model"] = mcfg["name"]
    g = members.groupby(["station", "time"], as_index=False)
    mean = g.agg(u=("u", "mean"), v=("v", "mean"), ws=("ws", "mean"), dist_km=("dist_km", "first"))
    # scalar speed is verified as the mean of member speeds; direction from the mean vector
    _, mean["wd"] = uv_to_speed_dir(mean["u"], mean["v"])
    mean["model"] = mcfg["name"]
    return mean, members


def build_matched(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs, stations = load_observations(cfg["obs"])
    period = cfg.get("period") or {}
    if period.get("start"):
        obs = obs[obs["time"] >= pd.Timestamp(period["start"])]
    if period.get("end"):
        obs = obs[obs["time"] <= pd.Timestamp(period["end"])]

    frames, member_frames = [], []
    for mcfg in cfg["models"]:
        ext, members = _extract_model(mcfg, stations)
        frames.append(ext)
        if members is not None:
            member_frames.append(members)
    model_long = pd.concat(frames, ignore_index=True)
    matched = model_long.merge(obs, on=["station", "time"], how="inner")
    matched = matched[np.isfinite(matched["ws"]) & np.isfinite(matched["ws_obs"])]

    if cfg.get("common_sample", True):
        n_models = len(cfg["models"])
        counts = matched.groupby(["station", "time"])["model"].nunique()
        keep = counts[counts == n_models].index
        matched = matched.set_index(["station", "time"]).loc[keep].reset_index()

    if matched.empty:
        raise ValueError("no matched obs-model pairs: check time zones, period and station coordinates")
    matched = matched.sort_values(["model", "station", "time"]).reset_index(drop=True)
    members = None
    if member_frames:
        keys = matched[["station", "time", "ws_obs"]].drop_duplicates(["station", "time"])
        members = pd.concat(member_frames, ignore_index=True).merge(keys, on=["station", "time"], how="inner")
    return matched, stations, members


def run(config_path) -> Path:
    cfg = load_config(config_path)
    out = Path(cfg["output"])
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    matched, stations, members = build_matched(cfg)
    models = [m["name"] for m in cfg["models"]]
    thresholds = cfg.get("thresholds_ms", [10.0, 14.0])

    st_cont, st_cat = scores_table(matched, models, thresholds, by_station=True)
    all_cont, all_cat = scores_table(matched, models, thresholds, by_station=False)

    skill_rows = []
    ref = cfg.get("reference")
    if ref and ref in models:
        wide = matched.pivot_table(index=["station", "time"], columns="model", values="ws").sort_index()
        obs_w = matched.drop_duplicates(["station", "time"]).set_index(["station", "time"])["ws_obs"]
        obs_w = obs_w.reindex(wide.index)
        boot = cfg.get("bootstrap", {})
        for m in models:
            if m == ref:
                continue
            row = {"model": m, "reference": ref}
            row.update(
                skill_vs_reference(
                    obs_w.values,
                    wide[m].values,
                    wide[ref].values,
                    n_boot=boot.get("n", 1000),
                    block=boot.get("block", 24),
                    seed=boot.get("seed", 0),
                )
            )
            skill_rows.append(row)
    skill = pd.DataFrame(skill_rows)

    matched.to_csv(out / "matched.csv", index=False)
    st_cont.to_csv(out / "scores_station.csv", index=False)
    st_cat.to_csv(out / "categorical_station.csv", index=False)
    all_cont.to_csv(out / "scores_all.csv", index=False)
    all_cat.to_csv(out / "categorical_all.csv", index=False)
    if not skill.empty:
        skill.to_csv(out / "skill_vs_reference.csv", index=False)

    plots.timeseries(matched, models, stations, figdir)
    plots.scatter(matched, models, figdir)
    plots.qq(matched, models, figdir)
    plots.score_bars(st_cont, models, figdir)

    prob = pd.DataFrame()
    if members is not None:
        prob, rel = ensemble_scores(matched, members, models, thresholds, ref)
        prob.to_csv(out / "probabilistic_all.csv", index=False)
        rel.to_csv(out / "reliability_table.csv", index=False)
        members.to_csv(out / "matched_members.csv", index=False)
        plots.rank_histograms(members, figdir)
        plots.reliability(rel, figdir)

    _write_summary(out, cfg, matched, all_cont, all_cat, skill, prob)
    with open(out / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dsverify_version": __version__,
                "case": cfg.get("case"),
                "models": models,
                "reference": ref,
                "n_pairs_per_model": int(len(matched) / len(models)),
                "stations": sorted(matched["station"].unique().tolist()),
                "time_start_utc": str(matched["time"].min()),
                "time_end_utc": str(matched["time"].max()),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return out


def _fmt(df: pd.DataFrame, cols) -> str:
    cols = [c for c in cols if c in df.columns]
    d = df[cols].copy()
    for c in cols:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda x: f"{x:.2f}")
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False))


def _write_summary(out, cfg, matched, all_cont, all_cat, skill, prob=None):
    lines = [
        f"# Verification summary: {cfg.get('case', '')}",
        "",
        f"- Period (UTC): {matched['time'].min()} to {matched['time'].max()}",
        f"- Stations: {matched['station'].nunique()}, pairs per model: {int(len(matched) / matched['model'].nunique())}",
        "",
        "## Continuous scores (all stations pooled, m/s)",
        "",
        _fmt(all_cont, ["model", "n", "bias", "mae", "rmse", "corr", "std_ratio", "p99_bias", "max_obs", "max_mod", "dir_mae", "vector_rmse"]),
        "",
        "## Strong-wind event scores",
        "",
        _fmt(all_cat, ["model", "threshold", "hits", "misses", "false_alarms", "pod", "far", "csi", "fbi"]),
        "",
    ]
    if not skill.empty:
        lines += [
            "## MSE skill score vs reference (moving-block bootstrap 95% CI)",
            "",
            _fmt(skill, ["model", "reference", "n", "msess", "msess_lo95", "msess_hi95", "p_not_better"]),
            "",
        ]
    if prob is not None and not prob.empty:
        lines += [
            "## Probabilistic scores (fair CRPS; deterministic models: CRPS = MAE)",
            "",
            _fmt(prob, ["model", "members", "n", "crps", "crps_fair", "crpss_vs_ref", "spread", "rmse_mean",
                        "spread_skill_ratio", "rank_delta", "outside_fraction", "expected_outside"]),
            "",
        ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
