"""Readers for station wind observations.

Supported formats
-----------------
kma_portal : hourly ASOS/AWS CSV downloaded from the KMA Open MET Data Portal
             (data.kma.go.kr). Korean headers, usually CP949 encoded.
api_json   : JSON saved from the data.go.kr ASOS hourly API (items with tm, stnId, ws, wd).
tidy       : any CSV with columns station, time, ws, wd (wd optional).

All readers return a tidy DataFrame with columns
    station (str), time (UTC, tz-naive datetime64), ws_obs (m/s), wd_obs (deg)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .wind import log_law_adjust

PORTAL_COLUMNS = {
    "station": ["지점", "지점번호", "stnId"],
    "time": ["일시", "tm"],
    "ws": ["풍속(m/s)", "풍속", "ws"],
    "wd": ["풍향(16방위)", "풍향(deg)", "풍향", "wd"],
}


def _read_csv_any_encoding(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("dsverify", b"", 0, 1, f"cannot decode {path}")


def _pick(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"none of {candidates} found in columns {list(df.columns)}")
    return None


def _to_utc_naive(times: pd.Series, timezone: str) -> pd.Series:
    t = pd.to_datetime(times)
    if t.dt.tz is None:
        t = t.dt.tz_localize(timezone)
    return t.dt.tz_convert("UTC").dt.tz_localize(None)


def read_kma_portal(path, timezone: str = "Asia/Seoul") -> pd.DataFrame:
    df = _read_csv_any_encoding(Path(path))
    c_st = _pick(df, PORTAL_COLUMNS["station"])
    c_t = _pick(df, PORTAL_COLUMNS["time"])
    c_ws = _pick(df, PORTAL_COLUMNS["ws"])
    c_wd = _pick(df, PORTAL_COLUMNS["wd"], required=False)
    out = pd.DataFrame(
        {
            "station": df[c_st].astype(str),
            "time": _to_utc_naive(df[c_t], timezone),
            "ws_obs": pd.to_numeric(df[c_ws], errors="coerce"),
            "wd_obs": pd.to_numeric(df[c_wd], errors="coerce") if c_wd else np.nan,
        }
    )
    return out


def read_api_json(path, timezone: str = "Asia/Seoul") -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        items = payload["response"]["body"]["items"]["item"]
    else:
        items = payload
    df = pd.DataFrame(items)
    df = df.replace("", np.nan)
    return pd.DataFrame(
        {
            "station": df["stnId"].astype(str),
            "time": _to_utc_naive(df["tm"], timezone),
            "ws_obs": pd.to_numeric(df["ws"], errors="coerce"),
            "wd_obs": pd.to_numeric(df.get("wd"), errors="coerce"),
        }
    )


def read_tidy(path, timezone: str = "UTC") -> pd.DataFrame:
    df = _read_csv_any_encoding(Path(path))
    return pd.DataFrame(
        {
            "station": df["station"].astype(str),
            "time": _to_utc_naive(df["time"], timezone),
            "ws_obs": pd.to_numeric(df["ws"], errors="coerce"),
            "wd_obs": pd.to_numeric(df["wd"], errors="coerce") if "wd" in df else np.nan,
        }
    )


READERS = {"kma_portal": read_kma_portal, "api_json": read_api_json, "tidy": read_tidy}


def read_stations(path) -> pd.DataFrame:
    """Station metadata CSV with columns station, name, lat, lon and optional height_m."""
    df = _read_csv_any_encoding(Path(path))
    df["station"] = df["station"].astype(str)
    if "height_m" not in df:
        df["height_m"] = np.nan
    return df[["station", "name", "lat", "lon", "height_m"]]


def load_observations(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load observations and station metadata according to the `obs` config block."""
    fmt = cfg.get("format", "kma_portal")
    tz = cfg.get("timezone", "Asia/Seoul" if fmt != "tidy" else "UTC")
    obs = READERS[fmt](cfg["path"], timezone=tz)
    stations = read_stations(cfg["station_meta"])

    if cfg.get("stations"):
        keep = [str(s) for s in cfg["stations"]]
        obs = obs[obs["station"].isin(keep)]
        stations = stations[stations["station"].isin(keep)]

    # basic QC: physical range, remove duplicates
    obs = obs[(obs["ws_obs"] >= 0) & (obs["ws_obs"] < 75)]
    obs.loc[(obs["wd_obs"] < 0) | (obs["wd_obs"] > 360), "wd_obs"] = np.nan
    obs = obs.drop_duplicates(["station", "time"])

    # optional height correction to 10 m
    if cfg.get("adjust_to_10m", False):
        h = obs["station"].map(stations.set_index("station")["height_m"])
        z0 = cfg.get("z0", 0.03)
        obs["ws_obs"] = [
            log_law_adjust(w, None if np.isnan(z) else z, 10.0, z0) for w, z in zip(obs["ws_obs"], h)
        ]
    return obs.reset_index(drop=True), stations.reset_index(drop=True)
