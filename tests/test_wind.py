import numpy as np
import pytest

from dsverify.wind import angular_difference, log_law_adjust, speed_dir_to_uv, uv_to_speed_dir


@pytest.mark.parametrize(
    "u,v,ws,wd",
    [(0, -5, 5, 0), (-5, 0, 5, 90), (0, 5, 5, 180), (5, 0, 5, 270)],  # N, E, S, W winds
)
def test_uv_to_speed_dir_convention(u, v, ws, wd):
    s, d = uv_to_speed_dir(u, v)
    assert s == pytest.approx(ws)
    assert d == pytest.approx(wd)


def test_round_trip():
    rng = np.random.default_rng(1)
    ws, wd = rng.uniform(0.5, 30, 100), rng.uniform(0, 360, 100)
    s, d = uv_to_speed_dir(*speed_dir_to_uv(ws, wd))
    assert np.allclose(s, ws)
    assert np.allclose(angular_difference(d, wd), 0, atol=1e-9)


def test_calm_direction_is_nan():
    _, d = uv_to_speed_dir(0.0, 0.0)
    assert np.isnan(d)


def test_angular_difference_wraps():
    assert angular_difference(10, 350) == pytest.approx(20)
    assert angular_difference(350, 10) == pytest.approx(-20)


def test_log_law_increases_to_10m():
    assert log_law_adjust(5.0, 2.0) > 5.0
    assert log_law_adjust(5.0, None) == pytest.approx(5.0)
