# Changelog

## 0.2.0 (2026-11-18)

- Ensemble verification: `member_dim` in a model config (default name `ensemble`, as written by Earth2Studio)
- New `dsverify.probabilistic`: CRPS (plain and fair), Gaussian CRPS, spread-skill ratio, rank histogram and
  flatness, Brier score with Murphy decomposition and reliability table
- New outputs: probabilistic_all.csv, reliability_table.csv, matched_members.csv, rank-histogram and
  reliability figures, probabilistic section in summary.md
- Deterministic models are scored with CRPS too (equals MAE) so all models share one proper score
- Synthetic demo now includes an 8-member CorrDiff-like ensemble

## 0.1.0 (2026-11-17)

- Station verification of downscaled 10 m wind: continuous, categorical and MSE skill scores with bootstrap CI
