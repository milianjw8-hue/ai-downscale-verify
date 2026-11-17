"""Fetch hourly ASOS observations from the data.go.kr open API and save as JSON.

Service: KMA ASOS hourly data (기상청_지상(종관, ASOS) 시간자료 조회서비스), data.go.kr.
Apply for the service on data.go.kr and put the decoded service key in the environment:

  export DATA_GO_KR_KEY="..."
  python scripts/fetch_asos_hourly.py --stations 184 185 188 189 \
      --start "2025-04-11 00" --end "2025-04-13 23" --out data/obs/asos_20250411-13.json

Times are KST. The saved file is read with `obs.format: api_json`.
Check the current API specification on data.go.kr if the request fails; parameters
and limits (e.g. maximum rows per page, latest available date) may change.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

URL = "http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"


def fetch_station(session, key, stn, start, end, rows=999):
    items, page = [], 1
    while True:
        params = {
            "serviceKey": key,
            "pageNo": page,
            "numOfRows": rows,
            "dataType": "JSON",
            "dataCd": "ASOS",
            "dateCd": "HR",
            "startDt": start[:8],
            "startHh": start[8:10],
            "endDt": end[:8],
            "endHh": end[8:10],
            "stnIds": stn,
        }
        r = session.get(URL, params=params, timeout=30)
        r.raise_for_status()
        body = r.json()["response"]["body"]
        batch = body["items"]["item"]
        items.extend(batch)
        if page * rows >= int(body["totalCount"]):
            return items
        page += 1
        time.sleep(0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", nargs="+", required=True)
    ap.add_argument("--start", required=True, help='"YYYY-MM-DD HH" in KST')
    ap.add_argument("--end", required=True, help='"YYYY-MM-DD HH" in KST')
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import requests

    key = os.environ.get("DATA_GO_KR_KEY")
    if not key:
        raise SystemExit("set DATA_GO_KR_KEY (never commit the key to git)")
    start = args.start.replace("-", "").replace(" ", "")
    end = args.end.replace("-", "").replace(" ", "")
    with requests.Session() as s:
        allitems = []
        for stn in args.stations:
            allitems += fetch_station(s, key, stn, start, end)
            print(f"station {stn}: total {len(allitems)} rows")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(allitems, f, ensure_ascii=False)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
