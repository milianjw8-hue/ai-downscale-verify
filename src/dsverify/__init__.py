"""dsverify: verify AI-downscaled near-surface wind against independent station observations.

Typical use compares a coarse reanalysis (e.g. ERA5), an AI downscaling product
(e.g. CorrDiff) and an operational NWP field against ASOS/AWS wind records.
"""

__version__ = "0.2.0"

from .wind import uv_to_speed_dir, speed_dir_to_uv  # noqa: F401
from .metrics import continuous_scores, categorical_scores, skill_vs_reference  # noqa: F401
