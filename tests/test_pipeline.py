import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from dsverify.pipeline import run

ROOT = Path(__file__).resolve().parents[1]


def test_end_to_end_synthetic(tmp_path):
    data = tmp_path / "data" / "demo"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_synthetic_demo.py"), "--out", str(data)], check=True
    )
    cfg = yaml.safe_load((ROOT / "configs" / "example_jeju_synthetic.yaml").read_text(encoding="utf-8"))
    cfg["root"] = str(tmp_path)
    cfg["bootstrap"]["n"] = 200
    cfg_path = tmp_path / "configs" / "cfg.yaml"
    cfg_path.parent.mkdir()
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    out = run(cfg_path)
    scores = pd.read_csv(out / "scores_all.csv").set_index("model")
    # the synthetic CorrDiff field is built to be closer to the truth than the smoothed ERA5 field
    assert scores.loc["CorrDiff", "rmse"] < scores.loc["ERA5", "rmse"]
    # KST observations must line up with UTC model times: all 36 hours x 4 stations matched
    assert scores.loc["ERA5", "n"] == 144
    skill = pd.read_csv(out / "skill_vs_reference.csv")
    assert skill.loc[0, "msess"] > 0
    assert (out / "figures" / "scatter.png").exists()
    # ensemble model: probabilistic table, CRPS = MAE for deterministic models, ensemble beats ERA5
    prob = pd.read_csv(out / "probabilistic_all.csv").set_index("model")
    assert prob.loc["CorrDiff-ENS", "members"] == 8
    assert abs(prob.loc["CorrDiff", "crps"] - scores.loc["CorrDiff", "mae"]) < 1e-9
    assert prob.loc["CorrDiff-ENS", "crps"] < prob.loc["ERA5", "crps"]
    assert 0.5 < prob.loc["CorrDiff-ENS", "spread_skill_ratio"] < 1.5
    assert (out / "figures" / "rank_histogram_CorrDiff-ENS.png").exists()
    assert (out / "reliability_table.csv").exists()
    assert (out / "summary.md").exists()
    shutil.rmtree(out)
