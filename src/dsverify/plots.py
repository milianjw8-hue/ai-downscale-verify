"""Standard verification figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]


def _color(models):
    return {m: COLORS[i % len(COLORS)] for i, m in enumerate(models)}


def timeseries(matched: pd.DataFrame, models, stations: pd.DataFrame, outdir: Path):
    col = _color(models)
    names = stations.set_index("station")["name"].to_dict()
    for st, g in matched.groupby("station"):
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ob = g[g["model"] == models[0]].sort_values("time")
        ax.plot(ob["time"], ob["ws_obs"], "k.-", lw=1.2, ms=4, label="OBS")
        for m in models:
            s = g[g["model"] == m].sort_values("time")
            ax.plot(s["time"], s["ws"], "-", color=col[m], lw=1.2, label=m)
        ax.set_ylabel("10 m wind speed (m s$^{-1}$)")
        ax.set_title(f"{names.get(st, st)} ({st})")
        ax.grid(alpha=0.3)
        ax.legend(ncol=len(models) + 1, fontsize=8, loc="upper left")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(outdir / f"timeseries_{st}.png", dpi=130)
        plt.close(fig)


def scatter(matched: pd.DataFrame, models, outdir: Path):
    col = _color(models)
    vmax = np.nanmax([matched["ws_obs"].max(), matched["ws"].max()]) * 1.05
    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 3.6), squeeze=False)
    for ax, m in zip(axes[0], models):
        s = matched[matched["model"] == m]
        ax.scatter(s["ws_obs"], s["ws"], s=8, alpha=0.5, color=col[m])
        ax.plot([0, vmax], [0, vmax], "k--", lw=0.8)
        ax.set_xlim(0, vmax)
        ax.set_ylim(0, vmax)
        ax.set_aspect("equal")
        ax.set_xlabel("OBS (m s$^{-1}$)")
        ax.set_ylabel(f"{m} (m s$^{-1}$)")
        ax.set_title(m)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "scatter.png", dpi=130)
    plt.close(fig)


def qq(matched: pd.DataFrame, models, outdir: Path):
    col = _color(models)
    q = np.linspace(0.01, 0.99, 99)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    base = matched[matched["model"] == models[0]]["ws_obs"].dropna()
    qo = np.quantile(base, q)
    for m in models:
        s = matched[matched["model"] == m]["ws"].dropna()
        ax.plot(qo, np.quantile(s, q), "o-", ms=2.5, color=col[m], label=m)
    vmax = max(qo.max(), matched["ws"].quantile(0.99)) * 1.05
    ax.plot([0, vmax], [0, vmax], "k--", lw=0.8)
    ax.set_xlabel("OBS quantile (m s$^{-1}$)")
    ax.set_ylabel("Model quantile (m s$^{-1}$)")
    ax.set_title("Q-Q plot (1-99th percentile)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "qq.png", dpi=130)
    plt.close(fig)


def score_bars(station_scores: pd.DataFrame, models, outdir: Path, keys=("bias", "rmse", "corr")):
    col = _color(models)
    stations = list(dict.fromkeys(station_scores["station"]))
    x = np.arange(len(stations))
    w = 0.8 / len(models)
    fig, axes = plt.subplots(len(keys), 1, figsize=(max(6, 1.2 * len(stations)), 2.6 * len(keys)), sharex=True)
    for ax, k in zip(np.atleast_1d(axes), keys):
        for i, m in enumerate(models):
            s = station_scores[station_scores["model"] == m].set_index("station").reindex(stations)
            ax.bar(x + (i - (len(models) - 1) / 2) * w, s[k], w, color=col[m], label=m)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel(k)
        ax.grid(axis="y", alpha=0.3)
    np.atleast_1d(axes)[0].legend(ncol=len(models), fontsize=8)
    np.atleast_1d(axes)[-1].set_xticks(x, stations)
    fig.tight_layout()
    fig.savefig(outdir / "station_scores.png", dpi=130)
    plt.close(fig)


def rank_histograms(members: pd.DataFrame, outdir: Path):
    """Rank histogram per ensemble model (flat = consistent; U-shape = too little spread)."""
    from .ensemble import member_array
    from .probabilistic import rank_flatness, rank_histogram

    for m in members["model"].unique():
        x, y, _ = member_array(members, m)
        c = rank_histogram(y, x)
        f = rank_flatness(c)
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        ax.bar(np.arange(len(c)), c / c.sum(), color="#1f77b4", width=0.85)
        ax.axhline(1 / len(c), color="k", ls="--", lw=0.8, label="flat")
        ax.set_xlabel("rank of observation among members")
        ax.set_ylabel("relative frequency")
        ax.set_title(f"{m}: delta={f['delta']:.2f}, outside={100 * f['outside_fraction']:.0f}%", fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(outdir / f"rank_histogram_{m}.png", dpi=130)
        plt.close(fig)


def reliability(rel: pd.DataFrame, outdir: Path):
    """Reliability diagram per threshold (forecast probability vs observed frequency)."""
    if rel.empty:
        return
    for thr, g in rel.groupby("threshold"):
        models = list(dict.fromkeys(g["model"]))
        col = _color(models)
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        for m in models:
            s = g[g["model"] == m]
            ax.plot(s["forecast_prob"], s["observed_freq"], "o-", color=col[m], ms=4, label=m)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("forecast probability")
        ax.set_ylabel("observed frequency")
        ax.set_title(f"wind speed >= {thr:g} m/s")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(outdir / f"reliability_{thr:g}.png", dpi=130)
        plt.close(fig)
