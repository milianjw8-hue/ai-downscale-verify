"""Generate a small SYNTHETIC Jeju strong-wind case for testing the pipeline.

Nothing here is real data. A hand-built "truth" wind field (synoptic flow modified by a
crude island-friction term and a west-tip acceleration near Gosan) is sampled to create:

  data/demo/era5_like.nc      0.25 deg regular grid, smoothed and slightly weakened
  data/demo/corrdiff_like.nc  ~1 km curvilinear grid (2-D lat/lon), truth + noise
  data/demo/asos_hourly.csv   4 stations, KMA-portal style CSV (CP949, KST)
  data/demo/stations.csv      station metadata (approximate coordinates)

Usage:  python scripts/make_synthetic_demo.py [--out data/demo] [--seed 42]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

# Approximate ASOS station coordinates; check official values on data.kma.go.kr before real use.
STATIONS = pd.DataFrame(
    {
        "station": ["184", "185", "188", "189"],
        "name": ["Jeju", "Gosan", "Seongsan", "Seogwipo"],
        "name_ko": ["제주", "고산", "성산", "서귀포"],
        "lat": [33.5141, 33.2938, 33.3868, 33.2462],
        "lon": [126.5297, 126.1628, 126.8802, 126.5653],
        "height_m": [np.nan] * 4,
    }
)


def truth_field(lat, lon, times):
    """Return (u, v) arrays (time, lat, lon) of the synthetic truth."""
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    # island friction: slowest at the summit, normalised elliptical distance r
    r2 = ((la - 33.36) / 0.17) ** 2 + ((lo - 126.53) / 0.40) ** 2
    island = np.exp(-r2 / 0.5)
    gosan_jet = np.exp(-((la - 33.29) ** 2 + (lo - 126.16) ** 2) / (2 * 0.04**2))
    factor = 1.0 - 0.45 * island + 0.35 * gosan_jet

    hours = (times - times[0]) / np.timedelta64(1, "h")
    peak = hours[len(hours) // 2]
    speed = 6.0 + 16.0 * np.exp(-((hours - peak) ** 2) / (2 * 6.0**2))  # m/s, peak 22
    direction = 200.0 + 70.0 * (hours / hours[-1])  # veering SSW -> W
    rad = np.radians(direction)
    u0, v0 = -speed * np.sin(rad), -speed * np.cos(rad)
    u = u0[:, None, None] * factor[None]
    v = v0[:, None, None] * factor[None]
    return u, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/demo")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    times = pd.date_range("2025-04-11T12:00", "2025-04-12T23:00", freq="1h").values  # UTC
    flat = np.round(np.arange(32.95, 33.80 + 1e-9, 0.01), 4)
    flon = np.round(np.arange(125.95, 127.30 + 1e-9, 0.01), 4)
    tu, tv = truth_field(flat, flon, times)
    # small-scale turbulence-like noise, spatially smooth
    for arr in (tu, tv):
        arr += np.stack([gaussian_filter(rng.normal(0, 3.0, arr.shape[1:]), 3) for _ in range(arr.shape[0])])

    # ERA5-like: 0.25 deg, heavy smoothing and 10 % low bias
    elat = np.arange(33.75, 32.99, -0.25)  # descending, like ERA5
    elon = np.arange(126.0, 127.26, 0.25)
    su = np.stack([gaussian_filter(x, 12) for x in tu]) * 0.9
    sv = np.stack([gaussian_filter(x, 12) for x in tv]) * 0.9
    eu = np.empty((len(times), len(elat), len(elon)))
    ev = np.empty_like(eu)
    pts = np.array(np.meshgrid(elat, elon, indexing="ij")).reshape(2, -1).T
    for t in range(len(times)):
        eu[t] = RegularGridInterpolator((flat, flon), su[t])(pts).reshape(len(elat), len(elon))
        ev[t] = RegularGridInterpolator((flat, flon), sv[t])(pts).reshape(len(elat), len(elon))
    xr.Dataset(
        {"u10": (("valid_time", "latitude", "longitude"), eu.astype("float32")),
         "v10": (("valid_time", "latitude", "longitude"), ev.astype("float32"))},
        coords={"valid_time": times, "latitude": elat, "longitude": elon},
        attrs={"title": "SYNTHETIC ERA5-like field for dsverify demo (not real data)"},
    ).to_netcdf(out / "era5_like.nc")

    # CorrDiff-like: ~1 km grid rotated 10 deg, 2-D lat/lon, truth + noise + small bias
    ny, nx = 56, 86
    y = (np.arange(ny) - ny / 2) * 1.0  # km
    x = (np.arange(nx) - nx / 2) * 1.0
    yy, xx = np.meshgrid(y, x, indexing="ij")
    th = np.radians(10.0)
    xr_, yr_ = xx * np.cos(th) - yy * np.sin(th), xx * np.sin(th) + yy * np.cos(th)
    lat2 = 33.36 + yr_ / 111.0
    lon2 = 126.55 + xr_ / (111.0 * np.cos(np.radians(33.36)))
    pts2 = np.column_stack([lat2.ravel(), lon2.ravel()])
    cu = np.empty((len(times), ny, nx), dtype="float32")
    cv = np.empty_like(cu)
    for t in range(len(times)):
        cu[t] = RegularGridInterpolator((flat, flon), tu[t])(pts2).reshape(ny, nx)
        cv[t] = RegularGridInterpolator((flat, flon), tv[t])(pts2).reshape(ny, nx)
    cu += rng.normal(0.2, 1.0, cu.shape).astype("float32")
    cv += rng.normal(0.0, 1.0, cv.shape).astype("float32")
    xr.Dataset(
        {"u10m": (("time", "y", "x"), cu), "v10m": (("time", "y", "x"), cv)},
        coords={"time": times, "lat": (("y", "x"), lat2), "lon": (("y", "x"), lon2)},
        attrs={"title": "SYNTHETIC CorrDiff-like field for dsverify demo (not real data)"},
    ).to_netcdf(out / "corrdiff_like.nc")

    # Observations: truth at stations + instrument/representativeness noise, KST, 16-point direction
    rows = []
    for _, st in STATIONS.iterrows():
        for t in range(len(times)):
            u = RegularGridInterpolator((flat, flon), tu[t])([st.lat, st.lon])[0] + rng.normal(0, 0.8)
            v = RegularGridInterpolator((flat, flon), tv[t])([st.lat, st.lon])[0] + rng.normal(0, 0.8)
            ws = float(np.hypot(u, v))
            wd = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
            wd16 = (np.round(wd / 22.5) * 22.5) % 360.0
            kst = pd.Timestamp(times[t]) + pd.Timedelta(hours=9)
            rows.append({"지점": st.station, "지점명": st.name_ko, "일시": kst.strftime("%Y-%m-%d %H:%M"),
                         "기온(°C)": np.nan, "풍속(m/s)": round(ws, 1), "풍향(16방위)": int(round(wd16))})
    pd.DataFrame(rows).to_csv(out / "asos_hourly.csv", index=False, encoding="cp949")
    STATIONS[["station", "name", "lat", "lon", "height_m"]].to_csv(out / "stations.csv", index=False)
    print(f"synthetic demo written to {out.resolve()}")


if __name__ == "__main__":
    main()
