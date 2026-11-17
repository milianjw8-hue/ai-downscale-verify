"""End-to-end verification run driven by a YAML config."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import __version__, plots
from .grid import extract_at_stations, open_model
from .metrics import scores_table, skill_vs_reference
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


def build_matched(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs, stations = load_observations(cfg["obs"])
    period = cfg.get("period") or {}
    if period.get("start"):
        obs = obs[obs["time"] >= pd.Timestamp(period["start"])]
    if period.get("end"):
        obs = obs[obs["time"] <= pd.Timestamp(period["end"])]

    frames = []
    for mcfg in cfg["models"]:
        ds = open_model(mcfg)
        ext = extract_at_stations(ds, stations, mcfg)
        ext["model"] = mcfg["name"]
        frames.append(ext)
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
    return matched.sort_values(["model", "station", "time"]).reset_index(drop=True), stations


def run(config_path) -> Path:
    cfg = load_config(config_path)
    out = Path(cfg["output"])
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    matched, stations = build_matched(cfg)
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

    _write_summary(out, cfg, matched, all_cont, all_cat, skill)
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


def _write_summary(out, cfg, matched, all_cont, all_cat, skill):
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
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
