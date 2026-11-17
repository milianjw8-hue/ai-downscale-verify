"""Extract gridded model wind at station locations.

Handles regular grids with 1-D latitude/longitude (e.g. ERA5) and curvilinear
grids with 2-D latitude/longitude (e.g. CorrDiff or LDAPS output on a projected grid).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

from .wind import uv_to_speed_dir

EARTH_RADIUS_KM = 6371.0


def _to_xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def _normalize_lon(lon, grid_lon):
    """Bring station longitudes to the grid's convention (0..360 or -180..180)."""
    if np.nanmax(grid_lon) > 180:
        return np.mod(lon, 360.0)
    return ((np.asarray(lon) + 180.0) % 360.0) - 180.0


def open_model(cfg: dict) -> xr.Dataset:
    paths = cfg["path"]
    if isinstance(paths, str) and any(ch in paths for ch in "*?["):
        ds = xr.open_mfdataset(paths, combine="by_coords")
    elif isinstance(paths, list):
        ds = xr.open_mfdataset(paths, combine="by_coords")
    else:
        ds = xr.open_dataset(paths)
    rename = {cfg.get("time", "time"): "time"}
    ds = ds.rename({k: v for k, v in rename.items() if k in ds.variables and k != v})
    if "level" in cfg and cfg["level"] is not None:
        ds = ds.sel({cfg.get("level_dim", "level"): cfg["level"]})
    return ds


def extract_at_stations(ds: xr.Dataset, stations: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return tidy DataFrame (station, time, u, v, ws, wd, dist_km) for one model.

    Parameters in cfg: u, v, lat, lon names; interp in {"nearest", "bilinear", "idw4"};
    max_dist_km for rejecting stations outside the domain.
    """
    u_name, v_name = cfg.get("u", "u10"), cfg.get("v", "v10")
    lat_name, lon_name = cfg.get("lat", "latitude"), cfg.get("lon", "longitude")
    method = cfg.get("interp", "bilinear")
    max_dist = float(cfg.get("max_dist_km", 50.0))

    lat2 = ds[lat_name]
    lon2 = ds[lon_name]
    st_lat = stations["lat"].to_numpy(float)
    st_lon = _normalize_lon(stations["lon"].to_numpy(float), lon2.values)

    if lat2.ndim == 1:
        u, v, dist = _extract_regular(ds, u_name, v_name, lat_name, lon_name, st_lat, st_lon, method)
    else:
        u, v, dist = _extract_curvilinear(ds, u_name, v_name, lat2.values, lon2.values, st_lat, st_lon, method)

    times = pd.to_datetime(ds["time"].values)
    rows = []
    for i, st in enumerate(stations["station"]):
        if dist[i] > max_dist:
            continue
        ws, wd = uv_to_speed_dir(u[:, i], v[:, i])
        rows.append(
            pd.DataFrame(
                {"station": st, "time": times, "u": u[:, i], "v": v[:, i], "ws": ws, "wd": wd, "dist_km": dist[i]}
            )
        )
    if not rows:
        raise ValueError("no station lies inside the model domain (check max_dist_km and coordinates)")
    return pd.concat(rows, ignore_index=True)


def _extract_regular(ds, u_name, v_name, lat_name, lon_name, st_lat, st_lon, method):
    pts_lat = xr.DataArray(st_lat, dims="station")
    pts_lon = xr.DataArray(st_lon, dims="station")
    sub = ds[[u_name, v_name]]
    if method == "nearest":
        sel = sub.sel({lat_name: pts_lat, lon_name: pts_lon}, method="nearest")
    else:  # bilinear (idw4 treated as bilinear on regular grids)
        sel = sub.interp({lat_name: pts_lat, lon_name: pts_lon}, method="linear")
    sel = sel.transpose("time", "station")
    glat = ds[lat_name].values
    glon = ds[lon_name].values
    near_lat = glat[np.abs(glat[None, :] - st_lat[:, None]).argmin(1)]
    near_lon = glon[np.abs(glon[None, :] - st_lon[:, None]).argmin(1)]
    dist = _great_circle_km(st_lat, st_lon, near_lat, near_lon)
    # outside the grid -> NaN from interp; mark as far away
    inside = (st_lat >= glat.min()) & (st_lat <= glat.max()) & (st_lon >= glon.min()) & (st_lon <= glon.max())
    dist = np.where(inside, dist, np.inf)
    return sel[u_name].values, sel[v_name].values, dist


def _extract_curvilinear(ds, u_name, v_name, glat, glon, st_lat, st_lon, method):
    tree = cKDTree(_to_xyz(glat.ravel(), glon.ravel()))
    k = 1 if method == "nearest" else 4
    chord, idx = tree.query(_to_xyz(st_lat, st_lon), k=k)
    chord = np.atleast_2d(chord.T).T if k == 1 else chord
    idx = np.atleast_2d(idx.T).T if k == 1 else idx
    dist_km = 2 * EARTH_RADIUS_KM * np.arcsin(np.clip(chord / 2, 0, 1))

    u_da, v_da = ds[u_name], ds[v_name]
    spatial = [d for d in u_da.dims if d != "time"]
    u_flat = u_da.transpose("time", *spatial).values.reshape(u_da.sizes["time"], -1)
    v_flat = v_da.transpose("time", *spatial).values.reshape(v_da.sizes["time"], -1)

    if k == 1:
        w = np.ones_like(dist_km)
    else:
        w = 1.0 / np.maximum(dist_km, 1e-6) ** 2
        w /= w.sum(axis=1, keepdims=True)
    u = np.einsum("tsk,sk->ts", u_flat[:, idx], w)
    v = np.einsum("tsk,sk->ts", v_flat[:, idx], w)
    return u, v, dist_km[:, 0]


def _great_circle_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
