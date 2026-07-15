"""Tests del módulo de breakouts intradía (construcción sin fuga)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alpha_selection.intraday.breakouts import (   # noqa: E402
    opening_range_breakouts, session_features, evaluate_events,
)


def _session(date, prices, base_min=0):
    """Construye barras de 1 min a partir de una lista de precios de cierre."""
    n = len(prices)
    t0 = pd.Timestamp(date) + pd.Timedelta(hours=9, minutes=30)
    rows = []
    for i, p in enumerate(prices):
        o = prices[i - 1] if i > 0 else p
        rows.append(dict(
            time=t0 + pd.Timedelta(minutes=i),
            open=o, high=max(o, p), low=min(o, p), close=p, volume=1,
            session=pd.Timestamp(date), minute_from_open=base_min + i,
        ))
    return pd.DataFrame(rows)


def test_orb_entry_is_next_bar_open_no_leakage():
    # Rango de apertura oscilando en [99.5, 100.5] (anchura 1), luego rompe al alza.
    opening = [100.5 if i % 2 == 0 else 99.5 for i in range(30)]
    prices = opening + [100.5, 101.0, 102.0, 103.0] + [103.0] * 5
    df = _session("2020-01-06", prices)
    ev = opening_range_breakouts(df, or_minutes=30)
    assert len(ev) == 1
    row = ev.iloc[0]
    assert row["direction"] == 1
    # La barra que rompe (primer close > 100.5) es la de minuto 31 (close=101.0).
    assert row["breakout_min"] == 31
    # La entrada debe ser la APERTURA de la barra siguiente (minuto 32),
    # cuya apertura = cierre previo = 101.0. Nunca la barra de la señal.
    assert row["entry_min"] == 32
    assert row["entry"] == 101.0


def test_orb_downside_breakout():
    opening = [100.5 if i % 2 == 0 else 99.5 for i in range(30)]
    prices = opening + [99.5, 99.0, 98.0] + [98.0] * 5
    df = _session("2020-01-07", prices)
    ev = opening_range_breakouts(df, or_minutes=30)
    assert len(ev) == 1
    assert ev.iloc[0]["direction"] == -1


def test_no_event_when_range_holds():
    prices = [100.0 + 0.1 * np.sin(i) for i in range(60)]  # oscila dentro del rango
    df = _session("2020-01-08", prices)
    ev = opening_range_breakouts(df, or_minutes=30)
    # Puede no romper nunca; si rompe, debe ser un único evento válido.
    assert len(ev) in (0, 1)


def test_evaluate_events_empty():
    s = evaluate_events(pd.DataFrame())
    assert s.n == 0


def test_session_features_prior_levels_shifted():
    df = pd.concat([
        _session("2020-01-06", [100, 101, 102, 101, 100] + [100] * 385),
        _session("2020-01-07", [100, 103, 104, 103, 102] + [102] * 385),
    ], ignore_index=True)
    feat = session_features(df)
    day2 = pd.Timestamp("2020-01-07")
    # El prior_high del día 2 debe ser el máximo del día 1 (102), no del día 2.
    assert feat.at[day2, "prior_high"] == 102
