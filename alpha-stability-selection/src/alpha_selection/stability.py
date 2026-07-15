"""Selección por estabilidad — un solo procedimiento, no un torneo.

Idea (Meinshausen & Bühlmann, 2010, *Stability Selection*):
  Reajusta un modelo disperso muchas veces sobre submuestras/bootstraps de
  discovery y mide con qué frecuencia se selecciona cada característica.
  La frecuencia de selección (¿aparece en el 80%+ de los bootstraps?) es el
  criterio, no el coeficiente ni el Sharpe OOS.

Dos lentes, un mismo run:
  1. ``elasticnet_selection_frequency`` — lente lineal (ElasticNet / logistic
     elastic-net). Selección = coeficiente no nulo tras estandarizar.
  2. ``shallow_tree_selection_frequency`` — lente no lineal. Árboles poco
     profundos (depth 2-3) sobre bootstraps; selección = la variable aparece
     en algún split (importancia > 0).

El resultado combinado se recorta por PARSIMONIA (reproducibilidad +
de-correlación) si sobreviven más de ``max_features``. Nunca por rendimiento.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

Task = Literal["classification", "regression"]
Aggregator = Literal["intersection", "union", "elasticnet", "tree"]


@dataclass
class StabilityConfig:
    """Configuración de UN procedimiento de selección (= una variante DSR)."""

    task: Task = "classification"
    n_bootstraps: int = 200
    subsample_fraction: float = 0.5           # submuestreo sin reemplazo (M&B)
    stability_threshold: float = 0.80         # π: umbral de frecuencia de selección
    max_features: int = 5                     # tope de parsimonia
    corr_threshold: float = 0.70              # de-correlación al recortar
    aggregator: Aggregator = "intersection"   # cómo combinar las dos lentes
    # ElasticNet
    l1_ratios: Sequence[float] = (0.3, 0.5, 0.7, 0.9)
    en_coef_eps: float = 1e-6                 # |coef| por encima => seleccionada
    en_cv: int = 5
    en_max_iter: int = 5000
    # Árboles
    tree_max_depth: int = 3                   # 2-3 recomendado
    tree_importance_eps: float = 1e-6
    tree_bootstrap_replace: bool = True       # 200 bootstraps con reemplazo
    # Reproducibilidad
    random_state: int = 42

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["l1_ratios"] = list(self.l1_ratios)
        return d


@dataclass
class StabilityResult:
    features: List[str]
    freq_elasticnet: Dict[str, float]
    freq_tree: Dict[str, float]
    freq_combined: Dict[str, float]           # media de las dos lentes
    stable_set: List[str]                     # supera umbral según aggregator
    final_set: List[str]                      # tras recorte por parsimonia
    trim_applied: bool
    config: StabilityConfig
    n_samples: int = 0
    dropped_for_parsimony: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "features": self.features,
            "freq_elasticnet": self.freq_elasticnet,
            "freq_tree": self.freq_tree,
            "freq_combined": self.freq_combined,
            "stable_set": self.stable_set,
            "final_set": self.final_set,
            "trim_applied": self.trim_applied,
            "dropped_for_parsimony": self.dropped_for_parsimony,
            "config": self.config.as_dict(),
        }


# --------------------------------------------------------------------------
# Lente lineal: ElasticNet / logistic elastic-net
# --------------------------------------------------------------------------
def elasticnet_selection_frequency(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: StabilityConfig,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Frecuencia de selección por ElasticNet sobre submuestras de discovery."""
    from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV

    rng = np.random.default_rng(cfg.random_state)
    features = list(X.columns)
    counts = np.zeros(len(features), dtype=float)
    n = len(X)
    m = max(2, int(round(cfg.subsample_fraction * n)))

    Xv = X.to_numpy(dtype=float)
    yv = y.to_numpy()
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)

    for b in range(cfg.n_bootstraps):
        idx = rng.choice(n, size=m, replace=False)
        Xb, yb = Xv[idx], yv[idx]
        wb = None if w is None else w[idx]

        # Estandarizar dentro del bootstrap (evita fuga entre folds).
        scaler = StandardScaler()
        Xs = scaler.fit_transform(Xb)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                if cfg.task == "classification":
                    if len(np.unique(yb)) < 2:
                        continue
                    model = LogisticRegressionCV(
                        Cs=10,
                        cv=min(cfg.en_cv, _min_class_count(yb)),
                        penalty="elasticnet",
                        solver="saga",
                        l1_ratios=list(cfg.l1_ratios),
                        max_iter=cfg.en_max_iter,
                        scoring="neg_log_loss",
                        n_jobs=1,
                    )
                    model.fit(Xs, yb, sample_weight=wb)
                    coef = np.abs(model.coef_).ravel()
                else:
                    model = ElasticNetCV(
                        l1_ratio=list(cfg.l1_ratios),
                        cv=cfg.en_cv,
                        max_iter=cfg.en_max_iter,
                        n_jobs=1,
                    )
                    model.fit(Xs, yb, sample_weight=wb)
                    coef = np.abs(model.coef_).ravel()
            except Exception:
                # Un bootstrap degenerado no debe tumbar el procedimiento.
                continue

        selected = coef > cfg.en_coef_eps
        counts[selected] += 1.0

    freq = counts / cfg.n_bootstraps
    return dict(zip(features, freq.tolist()))


def _min_class_count(y: np.ndarray) -> int:
    _, c = np.unique(y, return_counts=True)
    return max(2, int(c.min()))


# --------------------------------------------------------------------------
# Lente no lineal: árboles poco profundos sobre bootstraps
# --------------------------------------------------------------------------
def shallow_tree_selection_frequency(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: StabilityConfig,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Frecuencia con que cada variable aparece en un split de un árbol depth<=3."""
    rng = np.random.default_rng(cfg.random_state + 1)
    features = list(X.columns)
    counts = np.zeros(len(features), dtype=float)
    n = len(X)

    Xv = X.to_numpy(dtype=float)
    yv = y.to_numpy()
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)

    Tree = DecisionTreeClassifier if cfg.task == "classification" else DecisionTreeRegressor

    for b in range(cfg.n_bootstraps):
        if cfg.tree_bootstrap_replace:
            idx = rng.choice(n, size=n, replace=True)
        else:
            m = max(2, int(round(cfg.subsample_fraction * n)))
            idx = rng.choice(n, size=m, replace=False)
        Xb, yb = Xv[idx], yv[idx]
        wb = None if w is None else w[idx]

        if cfg.task == "classification" and len(np.unique(yb)) < 2:
            continue

        tree = Tree(
            max_depth=cfg.tree_max_depth,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        try:
            tree.fit(Xb, yb, sample_weight=wb)
        except Exception:
            continue

        used = tree.feature_importances_ > cfg.tree_importance_eps
        counts[used] += 1.0

    freq = counts / cfg.n_bootstraps
    return dict(zip(features, freq.tolist()))


# --------------------------------------------------------------------------
# Recorte por parsimonia (nunca por rendimiento)
# --------------------------------------------------------------------------
def _parsimony_trim(
    stable: List[str],
    freq_combined: Dict[str, float],
    X: pd.DataFrame,
    cfg: StabilityConfig,
) -> tuple[List[str], List[str]]:
    """Si sobreviven más de max_features, recorta por reproducibilidad + de-correlación.

    Orden: mayor frecuencia de selección primero (más reproducible). Se van
    aceptando variables mientras no estén demasiado correlacionadas con una ya
    aceptada de mayor rango. Esto es PARSIMONIA, no rendimiento OOS.
    """
    if len(stable) <= cfg.max_features:
        return stable, []

    ranked = sorted(stable, key=lambda f: freq_combined.get(f, 0.0), reverse=True)
    corr = X[ranked].corr().abs()

    kept: List[str] = []
    dropped: List[str] = []
    for f in ranked:
        if len(kept) >= cfg.max_features:
            dropped.append(f)
            continue
        # ¿colineal con algo ya aceptado (de mayor frecuencia)?
        if any(corr.loc[f, k] > cfg.corr_threshold for k in kept):
            dropped.append(f)
            continue
        kept.append(f)

    # Si la de-correlación dejó huecos por debajo del tope, rellena con las
    # siguientes más estables aunque estén correlacionadas (parsimonia dura).
    if len(kept) < cfg.max_features:
        for f in ranked:
            if len(kept) >= cfg.max_features:
                break
            if f not in kept:
                kept.append(f)
                if f in dropped:
                    dropped.remove(f)
    return kept, dropped


# --------------------------------------------------------------------------
# Procedimiento completo (una variante)
# --------------------------------------------------------------------------
def stability_select(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Optional[StabilityConfig] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> StabilityResult:
    """Corre el procedimiento completo UNA vez sobre discovery.

    Devuelve las frecuencias de selección de ambas lentes, el conjunto estable
    (según el agregador) y el conjunto final tras recorte por parsimonia.
    """
    cfg = cfg or StabilityConfig()
    if X.isnull().any().any():
        raise ValueError("X contiene NaNs; límpialos antes de la selección.")

    features = list(X.columns)
    freq_en = elasticnet_selection_frequency(X, y, cfg, sample_weight)
    freq_tr = shallow_tree_selection_frequency(X, y, cfg, sample_weight)
    freq_comb = {f: 0.5 * (freq_en[f] + freq_tr[f]) for f in features}

    pi = cfg.stability_threshold
    stable_en = {f for f in features if freq_en[f] >= pi}
    stable_tr = {f for f in features if freq_tr[f] >= pi}

    if cfg.aggregator == "intersection":
        stable = stable_en & stable_tr
    elif cfg.aggregator == "union":
        stable = stable_en | stable_tr
    elif cfg.aggregator == "elasticnet":
        stable = stable_en
    elif cfg.aggregator == "tree":
        stable = stable_tr
    else:  # pragma: no cover
        raise ValueError(f"aggregator desconocido: {cfg.aggregator}")

    stable_list = sorted(stable, key=lambda f: freq_comb[f], reverse=True)
    final, dropped = _parsimony_trim(stable_list, freq_comb, X, cfg)

    return StabilityResult(
        features=features,
        freq_elasticnet=freq_en,
        freq_tree=freq_tr,
        freq_combined=freq_comb,
        stable_set=stable_list,
        final_set=final,
        trim_applied=bool(dropped) or len(stable_list) > cfg.max_features,
        dropped_for_parsimony=dropped,
        config=cfg,
        n_samples=len(X),
    )
