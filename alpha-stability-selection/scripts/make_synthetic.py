#!/usr/bin/env python
"""Genera un panel sintético para smoke-test del motor de selección.

No es dato real: es un banco de pruebas con estructura CONOCIDA para verificar
que el motor recupera las señales de verdad y descarta el ruido.

Construimos las candidatas del registro. Solo unas pocas están realmente
ligadas al target; el resto es ruido (a veces correlacionado). Un motor sano
debe dejar estables las señales y descartar el ruido.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Candidatas activas del registro por defecto.
FEATURES = [
    "spy_mom_63", "spy_mom_252", "spy_ret_5", "spy_vol_20", "spy_vol_ratio_20_60",
    "cyc_minus_def_mom_63", "xlk_rs_spy_20", "xlf_rs_spy_20", "xle_rs_spy_20",
    "sector_ret_dispersion_20", "sector_mom_mean_63", "macro_yield_slope_10y2y",
    "macro_hy_oas", "macro_usd_broad_chg_20", "macro_vix_level", "macro_vix_chg_5",
]

# Las que de verdad mueven el target (el resto es ruido afortunado en potencia).
TRUE_SIGNAL = {
    "spy_mom_63": 0.9,
    "cyc_minus_def_mom_63": 0.8,
    "macro_hy_oas": -0.7,
    "macro_vix_chg_5": -0.6,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic_discovery.parquet")
    ap.add_argument("--n", type=int, default=1100, help="nº de días (~890 discovery + OOS)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n = args.n
    dates = pd.bdate_range("2019-01-01", periods=n)

    X = rng.standard_normal((n, len(FEATURES)))
    df = pd.DataFrame(X, columns=FEATURES)

    # Introduce colinealidad realista: mom_252 correlaciona con mom_63.
    df["spy_mom_252"] = 0.7 * df["spy_mom_63"] + 0.3 * rng.standard_normal(n)
    # vol_ratio correlaciona con vol_20.
    df["spy_vol_ratio_20_60"] = 0.6 * df["spy_vol_20"] + 0.4 * rng.standard_normal(n)

    logit = np.zeros(n)
    for f, w in TRUE_SIGNAL.items():
        logit += w * df[f].to_numpy()
    logit += 0.4 * rng.standard_normal(n)  # ruido: señal/ruido deliberadamente flojo
    p = 1.0 / (1.0 + np.exp(-logit))
    df["spy_tp_hit"] = (rng.uniform(size=n) < p).astype(int)

    df.insert(0, "date", dates)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix in (".parquet", ".pq"):
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)
    print(f"Escrito {out}  shape={df.shape}")
    print(f"Señales verdaderas: {list(TRUE_SIGNAL)}")
    print(f"Tasa target: {df['spy_tp_hit'].mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
