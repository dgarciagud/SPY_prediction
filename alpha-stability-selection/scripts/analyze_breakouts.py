#!/usr/bin/env python
"""Análisis reproducible de breakouts intradía en SPX500 (datos FutureSharks/Oanda).

Discovery = sesiones < 2017-01-01. OOS 2017+ SELLADO (no se toca aquí).

Corre:
  1. Breakout incondicional (ORB 15/30/60, prior-day H/L), gross y neto de coste.
  2. Cortes condicionales preespecificados sobre ORB-30 (hipótesis económicas).
  3. Robustez de la hipótesis de continuación del gap (por año, ex-2008, neto).

No selecciona parámetros por rendimiento OOS. Imprime la evidencia; las
conclusiones van en intraday/FINDINGS.md.

Uso:
  python scripts/analyze_breakouts.py --cache data/spx_rth.parquet
Para (re)construir la caché desde los CSV de FutureSharks:
  python scripts/analyze_breakouts.py --spx-dir <.../SPX500_USD> --cache data/spx_rth.parquet --build
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alpha_selection.intraday.loader import build_rth_cache, load_rth_cache  # noqa: E402
from alpha_selection.intraday.breakouts import (                             # noqa: E402
    opening_range_breakouts, prior_day_breakouts, session_features,
    evaluate_events, stats_by_year,
)

OOS_START = pd.Timestamp("2017-01-01")
COST_PTS = 1.0


def _t(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    sd = x.std(ddof=1)
    return float(x.mean() / (sd / np.sqrt(len(x)))) if sd > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ROOT / "data/spx_rth.parquet"))
    ap.add_argument("--spx-dir", default=None)
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()

    if args.build:
        if not args.spx_dir:
            raise SystemExit("--build requiere --spx-dir con los CSV de SPX500_USD")
        print("Construyendo caché RTH...")
        build_rth_cache(args.spx_dir, args.cache)

    rth = load_rth_cache(args.cache)
    disc = rth[rth["session"] < OOS_START]
    print(f"DISCOVERY: {disc['session'].nunique()} sesiones "
          f"({disc['session'].min().date()} -> {disc['session'].max().date()})")
    print(f"OOS sellado: {rth[rth['session']>=OOS_START]['session'].nunique()} sesiones (>= {OOS_START.date()})")

    print("\n== 1. BREAKOUT INCONDICIONAL ==")
    for orm in (15, 30, 60):
        ev = opening_range_breakouts(disc, or_minutes=orm)
        s, sc = evaluate_events(ev, 0.0), evaluate_events(ev, COST_PTS)
        print(f"ORB-{orm:<3} n={s.n} L/S={s.n_long}/{s.n_short}  "
              f"gross={s.mean_ret_bps:+.2f}bps t={s.t_stat:+.2f} hit={s.hit_rate:.3f} "
              f"ft={s.follow_through_rate:.3f} | net(1pt)={sc.mean_ret_bps:+.2f}bps t={sc.t_stat:+.2f}")
    ev_pd = prior_day_breakouts(disc)
    s, sc = evaluate_events(ev_pd, 0.0), evaluate_events(ev_pd, COST_PTS)
    print(f"PDHL    n={s.n} L/S={s.n_long}/{s.n_short}  "
          f"gross={s.mean_ret_bps:+.2f}bps t={s.t_stat:+.2f} | net(1pt)={sc.mean_ret_bps:+.2f}bps t={sc.t_stat:+.2f}")

    print("\n== 2. CORTES CONDICIONALES (ORB-30, hipótesis preespecificadas) ==")
    ev = opening_range_breakouts(disc, or_minutes=30).copy()
    feat = session_features(disc)
    ev = ev.join(feat[["gap", "prior_close"]], on="session")
    ev["ret_bps"] = ev["ret_to_close"] * 1e4
    ev["aligned"] = (ev["direction"] == np.sign(ev["gap"])).astype(int)

    def cut(name, mask):
        g = ev.loc[mask, "ret_bps"]
        tag = "" if len(g) >= 100 else "  (muestra pequeña)"
        print(f"  {name:<30} n={len(g):<5} mean={g.mean():+6.2f}bps hit={(g>0).mean():.3f} t={_t(g):+.2f}{tag}")

    print(" H: continuación del gap")
    cut("dir == signo(gap)", ev["aligned"] == 1)
    cut("dir != signo(gap)", ev["aligned"] == 0)

    print("\n== 3. ROBUSTEZ: continuación del gap ==")
    al = ev[ev["aligned"] == 1].copy()
    al["year"] = pd.to_datetime(al["session"]).dt.year
    al["net_bps"] = (al["ret_to_close"] - COST_PTS / al["entry"]) * 1e4
    no08 = al[al["year"] != 2008]
    print(f"  gross={al['ret_bps'].mean():+.2f}bps t={_t(al['ret_bps']):+.2f} | "
          f"net(1pt)={al['net_bps'].mean():+.2f}bps t={_t(al['net_bps']):+.2f}")
    print(f"  EX-2008 gross={no08['ret_bps'].mean():+.2f}bps t={_t(no08['ret_bps']):+.2f} | "
          f"net={no08['net_bps'].mean():+.2f}bps t={_t(no08['net_bps']):+.2f}")
    by = al.groupby("year")["ret_bps"].agg(["size", "mean"]).round(2)
    print(f"  años gross>0: {(by['mean']>0).sum()}/{len(by)}")

    print("\nVEREDICTO: ni el incondicional ni la mejor hipótesis condicional")
    print("sobreviven a costes. OOS permanece sellado (no hay candidato que validar).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
