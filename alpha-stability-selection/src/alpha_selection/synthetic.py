"""Panel sintético de discovery para demostrar el motor de selección.

No es dato real: es un banco de pruebas con estructura CONOCIDA para verificar
que el motor recupera las señales de verdad y descarta el ruido. Solo unas pocas
candidatas están ligadas al target; el resto es ruido (a veces correlacionado).
Un motor sano deja estables las señales y descarta el ruido.
"""

from __future__ import annotations

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


def make_synthetic_discovery(
    n: int = 1100,
    seed: int = 7,
    target_name: str = "spy_tp_hit",
    start: str = "2019-01-01",
) -> pd.DataFrame:
    """Genera un panel diario sintético: date + candidatas + target binario.

    Introduce colinealidad realista (mom_252~mom_63, vol_ratio~vol_20) y un
    ratio señal/ruido deliberadamente flojo, para que el bootstrap de estabilidad
    tenga que trabajar.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)

    X = rng.standard_normal((n, len(FEATURES)))
    df = pd.DataFrame(X, columns=FEATURES)

    # Colinealidad realista.
    df["spy_mom_252"] = 0.7 * df["spy_mom_63"] + 0.3 * rng.standard_normal(n)
    df["spy_vol_ratio_20_60"] = 0.6 * df["spy_vol_20"] + 0.4 * rng.standard_normal(n)

    logit = np.zeros(n)
    for f, w in TRUE_SIGNAL.items():
        logit += w * df[f].to_numpy()
    logit += 0.4 * rng.standard_normal(n)  # ruido: señal/ruido flojo a propósito
    p = 1.0 / (1.0 + np.exp(-logit))
    df[target_name] = (rng.uniform(size=n) < p).astype(int)

    df.insert(0, "date", dates)
    return df
