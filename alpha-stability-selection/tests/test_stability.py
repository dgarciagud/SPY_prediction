"""Tests del motor de selección por estabilidad."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alpha_selection.splits import DataSplit, SealError            # noqa: E402
from alpha_selection.stability import StabilityConfig, stability_select  # noqa: E402
from alpha_selection.variants import VariantLog, log_variant       # noqa: E402


def _make_data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n)
    X = pd.DataFrame(rng.standard_normal((n, 8)),
                     columns=[f"f{i}" for i in range(8)])
    # f0 y f2 son señal; el resto ruido.
    logit = 1.2 * X["f0"] - 1.0 * X["f2"] + 0.3 * rng.standard_normal(n)
    y = pd.Series((rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int),
                  name="y")
    df = X.copy()
    df["y"] = y.values
    df.insert(0, "date", dates)
    return df


def test_seal_blocks_oos_access():
    df = _make_data()
    split = DataSplit(df, oos_start="2020-06-01", date_col="date")
    assert split.n_discovery > 0 and split.n_oos > 0
    with pytest.raises(SealError):
        _ = split.oos


def test_break_seal_logs(tmp_path):
    df = _make_data()
    log = tmp_path / "oos_access.jsonl"
    split = DataSplit(df, oos_start="2020-06-01", date_col="date", access_log=log)
    oos = split.break_seal(reason="gate final")
    assert len(oos) == split.n_oos
    assert log.exists()
    assert "gate final" in log.read_text()


def test_selection_recovers_signal_drops_noise():
    df = _make_data(n=700, seed=1)
    X = df[[c for c in df.columns if c.startswith("f")]]
    y = df["y"]
    cfg = StabilityConfig(n_bootstraps=40, stability_threshold=0.7, max_features=5)
    res = stability_select(X, y, cfg)
    # Las señales verdaderas deben estar entre las más frecuentes.
    assert res.freq_combined["f0"] > res.freq_combined["f1"]
    assert res.freq_combined["f2"] > res.freq_combined["f3"]
    # Al menos una señal verdadera sobrevive.
    assert "f0" in res.stable_set or "f2" in res.stable_set


def test_parsimony_trim_caps_features():
    df = _make_data(n=700, seed=2)
    X = df[[c for c in df.columns if c.startswith("f")]]
    y = df["y"]
    # Umbral bajísimo => todas "estables" => debe recortar a max_features.
    cfg = StabilityConfig(n_bootstraps=20, stability_threshold=0.0,
                          max_features=3, aggregator="union")
    res = stability_select(X, y, cfg)
    assert len(res.final_set) <= 3
    assert res.trim_applied


def test_variant_log_counts(tmp_path):
    df = _make_data(n=500)
    X = df[[c for c in df.columns if c.startswith("f")]]
    y = df["y"]
    cfg = StabilityConfig(n_bootstraps=15)
    res = stability_select(X, y, cfg)
    log_path = tmp_path / "variants.jsonl"
    r1 = log_variant(res, log_path, data_fingerprint="abc", label="v1")
    r2 = log_variant(res, log_path, data_fingerprint="abc", label="v2")
    assert r1["variant_index"] == 1
    assert r2["variant_index"] == 2
    assert VariantLog(log_path).count() == 2


def test_nan_rejected():
    df = _make_data(n=300)
    X = df[[c for c in df.columns if c.startswith("f")]].copy()
    X.iloc[0, 0] = np.nan
    y = df["y"]
    with pytest.raises(ValueError):
        stability_select(X, y, StabilityConfig(n_bootstraps=5))
