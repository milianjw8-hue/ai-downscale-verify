import numpy as np
import pytest

from dsverify.metrics import categorical_scores, continuous_scores, skill_vs_reference


def test_perfect_forecast():
    o = np.linspace(1, 20, 50)
    s = continuous_scores(o, o)
    assert s["bias"] == pytest.approx(0)
    assert s["rmse"] == pytest.approx(0)
    assert s["corr"] == pytest.approx(1)


def test_constant_bias():
    o = np.linspace(1, 20, 50)
    s = continuous_scores(o, o + 2)
    assert s["bias"] == pytest.approx(2)
    assert s["rmse"] == pytest.approx(2)
    assert s["p99_bias"] == pytest.approx(2)


def test_nan_pairs_dropped():
    s = continuous_scores([1, 2, np.nan, 4], [1, np.nan, 3, 4])
    assert s["n"] == 2


def test_direction_scores_ignore_calm():
    o = np.array([0.2, 5, 5, 5])
    s = continuous_scores(o, o, [90, 90, 90, 90], [270, 100, 80, 90])
    assert s["n_dir"] == 3
    assert s["dir_mae"] == pytest.approx(20 / 3)


def test_contingency():
    o = np.array([15, 15, 5, 5, 15])
    m = np.array([15, 5, 15, 5, 15])
    c = categorical_scores(o, m, 14)
    assert (c["hits"], c["misses"], c["false_alarms"], c["correct_negatives"]) == (2, 1, 1, 1)
    assert c["pod"] == pytest.approx(2 / 3)
    assert c["far"] == pytest.approx(1 / 3)
    assert c["csi"] == pytest.approx(0.5)
    assert c["fbi"] == pytest.approx(1.0)


def test_skill_better_than_reference():
    rng = np.random.default_rng(0)
    o = rng.uniform(2, 20, 240)
    good = o + rng.normal(0, 0.5, o.size)
    bad = o + rng.normal(0, 3.0, o.size)
    s = skill_vs_reference(o, good, bad, n_boot=300)
    assert s["msess"] > 0.8
    assert s["msess_lo95"] > 0
    assert s["p_not_better"] == 0
