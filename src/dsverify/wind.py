"""Wind vector utilities (meteorological convention).

Direction is the direction the wind blows FROM, in degrees clockwise from north.
"""

from __future__ import annotations

import numpy as np

DEFAULT_Z0 = 0.03  # m, open flat terrain (WMO class 1), used as default roughness


def uv_to_speed_dir(u, v):
    """Return (speed, direction_from_deg) from u (eastward) and v (northward)."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    speed = np.hypot(u, v)
    direction = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    direction = np.where(speed == 0, np.nan, direction)
    return speed, direction


def speed_dir_to_uv(speed, direction):
    """Return (u, v) from speed and direction-from in degrees."""
    speed = np.asarray(speed, dtype=float)
    rad = np.radians(np.asarray(direction, dtype=float))
    return -speed * np.sin(rad), -speed * np.cos(rad)


def angular_difference(a, b):
    """Signed smallest difference a - b in degrees, in [-180, 180)."""
    d = (np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + 180.0) % 360.0 - 180.0
    return d


def log_law_adjust(speed, z_from, z_to=10.0, z0=DEFAULT_Z0):
    """Adjust wind speed between heights with the neutral logarithmic profile.

    Use only as a first-order correction for anemometers not mounted at 10 m.
    """
    if z_from is None or z_from == z_to:
        return np.asarray(speed, dtype=float)
    if z_from <= z0 or z_to <= z0:
        raise ValueError("heights must exceed roughness length z0")
    return np.asarray(speed, dtype=float) * np.log(z_to / z0) / np.log(z_from / z0)
