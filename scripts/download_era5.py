"""Download ERA5 10 m wind for a case from the Copernicus Climate Data Store.

Requires a CDS account and ~/.cdsapirc (see https://cds.climate.copernicus.eu/how-to-api).
Install:  pip install "dsverify[download]"

Example (Jeju strong wind, 11-12 April 2025):
  python scripts/download_era5.py --start 2025-04-11 --end 2025-04-12 \
      --area 34.0 125.8 33.0 127.2 --out data/era5/era5_u10v10_20250411-12.nc
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--area", nargs=4, type=float, metavar=("N", "W", "S", "E"), default=[34.0, 125.8, 33.0, 127.2])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import cdsapi  # imported here so the core package does not depend on it

    days = pd.date_range(args.start, args.end, freq="D")
    request = {
        "product_type": ["reanalysis"],
        "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
        "year": sorted({d.strftime("%Y") for d in days}),
        "month": sorted({d.strftime("%m") for d in days}),
        "day": sorted({d.strftime("%d") for d in days}),
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": args.area,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cdsapi.Client().retrieve("reanalysis-era5-single-levels", request).download(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
