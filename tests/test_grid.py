import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dsverify.grid import extract_at_stations

TIMES = pd.date_range("2025-01-01", periods=3, freq="1h")
STATIONS = pd.DataFrame({"station": ["A", "B"], "name": ["A", "B"], "lat": [33.3, 33.45], "lon": [126.3, 126.7]})


def linear_field(lat, lon):
    # u depends linearly on lon, v on lat -> bilinear interpolation must be exact
    return 2.0 * (lon - 126.0), 3.0 * (lat - 33.0)


def test_regular_bilinear_exact():
    lat = np.arange(33.75, 32.99, -0.25)
    lon = np.arange(126.0, 127.01, 0.25)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    u, v = linear_field(la, lo)
    ds = xr.Dataset(
        {"u10": (("time", "latitude", "longitude"), np.repeat(u[None], 3, 0)),
         "v10": (("time", "latitude", "longitude"), np.repeat(v[None], 3, 0))},
        coords={"time": TIMES, "latitude": lat, "longitude": lon},
    )
    out = extract_at_stations(ds, STATIONS, {"interp": "bilinear"})
    a = out[out.station == "A"].iloc[0]
    assert a.u == pytest.approx(2.0 * 0.3)
    assert a.v == pytest.approx(3.0 * 0.3)


def test_curvilinear_nearest_and_domain_check():
    y, x = np.meshgrid(np.arange(20), np.arange(30), indexing="ij")
    lat2 = 33.2 + y * 0.02
    lon2 = 126.2 + x * 0.02
    u, v = linear_field(lat2, lon2)
    ds = xr.Dataset(
        {"u": (("time", "y", "x"), np.repeat(u[None], 3, 0)), "v": (("time", "y", "x"), np.repeat(v[None], 3, 0))},
        coords={"time": TIMES, "lat": (("y", "x"), lat2), "lon": (("y", "x"), lon2)},
    )
    far = pd.concat([STATIONS, pd.DataFrame({"station": ["FAR"], "name": ["FAR"], "lat": [35.0], "lon": [129.0]})])
    out = extract_at_stations(ds, far, {"u": "u", "v": "v", "lat": "lat", "lon": "lon", "interp": "nearest", "max_dist_km": 3})
    assert set(out.station) == {"A", "B"}
    a = out[out.station == "A"].iloc[0]
    assert a.u == pytest.approx(2.0 * 0.3, abs=0.03)
    assert a.dist_km < 1.5
