"""Test del generador de datos sintéticos de discovery."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alpha_selection.synthetic import make_synthetic_discovery, FEATURES, TRUE_SIGNAL  # noqa: E402


def test_shape_and_columns():
    df = make_synthetic_discovery(n=500, seed=1)
    assert len(df) == 500
    assert df.columns[0] == "date"
    for f in FEATURES:
        assert f in df.columns
    assert "spy_tp_hit" in df.columns
    assert set(df["spy_tp_hit"].unique()) <= {0, 1}


def test_deterministic_by_seed():
    a = make_synthetic_discovery(n=300, seed=42)
    b = make_synthetic_discovery(n=300, seed=42)
    assert a.equals(b)


def test_true_signals_are_features():
    for f in TRUE_SIGNAL:
        assert f in FEATURES
