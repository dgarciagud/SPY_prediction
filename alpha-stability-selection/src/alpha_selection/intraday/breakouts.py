"""Setups de breakout intradía y su evaluación honesta.

Construcción sin fuga de información:
  - La SEÑAL se confirma con el CIERRE de la barra que rompe el nivel.
  - La ENTRADA es la APERTURA de la barra siguiente (nunca la misma barra).
  - La SALIDA por defecto es el cierre de la sesión (RTH) — un breakout
    intradía se liquida antes de la campana.

Dos setups:
  1. Opening Range Breakout (ORB): rango de apertura = high/low de los primeros
     R minutos; primer cierre fuera del rango dispara el evento.
  2. Prior-day high/low breakout: primer cierre por encima del máximo (o por
     debajo del mínimo) RTH de la sesión anterior.

Evaluación por evento: retorno a cierre, MFE/MAE, y si un objetivo simétrico
(k x anchura) se alcanza antes que el stop (follow-through vs fade).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


def session_features(rth: pd.DataFrame) -> pd.DataFrame:
    """Agregados por sesión + niveles de la sesión previa (para prior-day)."""
    g = rth.groupby("session", sort=True)
    feat = g.agg(
        sess_open=("open", "first"),
        sess_close=("close", "last"),
        sess_high=("high", "max"),
        sess_low=("low", "min"),
        n_bars=("close", "size"),
    )
    feat["prior_high"] = feat["sess_high"].shift(1)
    feat["prior_low"] = feat["sess_low"].shift(1)
    feat["prior_close"] = feat["sess_close"].shift(1)
    feat["gap"] = feat["sess_open"] - feat["prior_close"]
    return feat


def _eval_path(direction, entry, post_high, post_low, post_close,
               width, target_k, stop_k):
    """Resultado de un evento dado el camino posterior a la entrada.

    Devuelve (ret_to_close, mfe, mae, hit_target_first).
    ret_to_close en fracción; mfe/mae en puntos a favor/en contra.
    hit_target_first: 1 si el objetivo se toca antes que el stop, 0 si al revés,
    np.nan si no se toca ninguno antes del cierre.
    """
    last_close = post_close[-1]
    ret_to_close = direction * (last_close - entry) / entry

    # Excursión favorable/adversa (en puntos, en la dirección del trade)
    if direction == 1:
        mfe = float(np.max(post_high) - entry)
        mae = float(entry - np.min(post_low))
    else:
        mfe = float(entry - np.min(post_low))
        mae = float(np.max(post_high) - entry)

    hit = np.nan
    if width > 0 and target_k > 0 and stop_k > 0:
        tgt = target_k * width
        stp = stop_k * width
        for i in range(len(post_close)):
            if direction == 1:
                up = post_high[i] - entry
                dn = entry - post_low[i]
            else:
                up = entry - post_low[i]
                dn = post_high[i] - entry
            hit_t = up >= tgt
            hit_s = dn >= stp
            if hit_t and hit_s:
                hit = 0  # ambos en la misma barra: conservador -> stop primero
                break
            if hit_t:
                hit = 1
                break
            if hit_s:
                hit = 0
                break
    return ret_to_close, mfe, mae, hit


def opening_range_breakouts(
    rth: pd.DataFrame,
    or_minutes: int = 30,
    target_k: float = 1.0,
    stop_k: float = 1.0,
    min_or_width: float = 0.0,
) -> pd.DataFrame:
    """Eventos ORB con evaluación. Un evento por sesión como máximo (primer toque)."""
    rows: List[dict] = []
    for session, s in rth.groupby("session", sort=True):
        s = s.sort_values("minute_from_open")
        m = s["minute_from_open"].to_numpy()
        o = s["open"].to_numpy(); h = s["high"].to_numpy()
        l = s["low"].to_numpy(); c = s["close"].to_numpy()

        or_mask = m < or_minutes
        if or_mask.sum() < 2:
            continue
        or_high = float(h[or_mask].max())
        or_low = float(l[or_mask].min())
        width = or_high - or_low
        if width <= min_or_width:
            continue

        post = np.where(m >= or_minutes)[0]
        direction = 0
        sig = None
        for i in post:
            if c[i] > or_high:
                direction, sig = 1, i
                break
            if c[i] < or_low:
                direction, sig = -1, i
                break
        if direction == 0 or sig is None or sig + 1 >= len(c):
            continue

        entry_idx = sig + 1
        entry = float(o[entry_idx])
        ph, pl, pc = h[entry_idx:], l[entry_idx:], c[entry_idx:]
        ret, mfe, mae, hit = _eval_path(direction, entry, ph, pl, pc,
                                        width, target_k, stop_k)
        rows.append(dict(
            session=session, setup="ORB", or_minutes=or_minutes,
            direction=direction, breakout_min=int(m[sig]),
            entry_min=int(m[entry_idx]), entry=entry,
            level=or_high if direction == 1 else or_low,
            width=width, ret_to_close=ret, mfe=mfe, mae=mae,
            hit_target_first=hit,
        ))
    return pd.DataFrame(rows)


def prior_day_breakouts(
    rth: pd.DataFrame,
    target_k: float = 1.0,
    stop_k: float = 1.0,
) -> pd.DataFrame:
    """Eventos de ruptura del máximo/mínimo RTH de la sesión anterior."""
    feat = session_features(rth)
    rows: List[dict] = []
    for session, s in rth.groupby("session", sort=True):
        if session not in feat.index:
            continue
        ph_level = feat.at[session, "prior_high"]
        pl_level = feat.at[session, "prior_low"]
        if not np.isfinite(ph_level) or not np.isfinite(pl_level):
            continue
        width = float(ph_level - pl_level)
        if width <= 0:
            continue

        s = s.sort_values("minute_from_open")
        m = s["minute_from_open"].to_numpy()
        o = s["open"].to_numpy(); h = s["high"].to_numpy()
        l = s["low"].to_numpy(); c = s["close"].to_numpy()

        direction = 0
        sig = None
        for i in range(len(c)):
            if c[i] > ph_level:
                direction, sig = 1, i
                break
            if c[i] < pl_level:
                direction, sig = -1, i
                break
        if direction == 0 or sig is None or sig + 1 >= len(c):
            continue

        entry_idx = sig + 1
        entry = float(o[entry_idx])
        ph, pl, pc = h[entry_idx:], l[entry_idx:], c[entry_idx:]
        ret, mfe, mae, hit = _eval_path(direction, entry, ph, pl, pc,
                                        width, target_k, stop_k)
        rows.append(dict(
            session=session, setup="PDHL",
            direction=direction, breakout_min=int(m[sig]),
            entry_min=int(m[entry_idx]), entry=entry,
            level=ph_level if direction == 1 else pl_level,
            width=width, ret_to_close=ret, mfe=mfe, mae=mae,
            hit_target_first=hit,
        ))
    return pd.DataFrame(rows)


@dataclass
class EventStats:
    n: int
    n_long: int
    n_short: int
    mean_ret_bps: float
    median_ret_bps: float
    hit_rate: float          # fracción con ret_to_close > 0
    t_stat: float
    follow_through_rate: float   # objetivo antes que stop (donde aplica)
    mean_mfe_pts: float
    mean_mae_pts: float
    fade_mean_bps: float     # retorno de hacer lo CONTRARIO (mean-reversion check)


def _cost_adjust(ret: np.ndarray, entry: np.ndarray, cost_pts: float) -> np.ndarray:
    """Resta coste round-trip (en puntos de índice) del retorno fraccional."""
    if cost_pts <= 0:
        return ret
    return ret - (cost_pts / entry)


def evaluate_events(events: pd.DataFrame, cost_pts: float = 0.0) -> EventStats:
    """Estadísticos agregados de un conjunto de eventos (después de coste)."""
    if len(events) == 0:
        return EventStats(0, 0, 0, np.nan, np.nan, np.nan, np.nan, np.nan,
                          np.nan, np.nan, np.nan)
    ret = _cost_adjust(events["ret_to_close"].to_numpy(),
                       events["entry"].to_numpy(), cost_pts)
    n = len(ret)
    sd = ret.std(ddof=1)
    t = float(ret.mean() / (sd / np.sqrt(n))) if sd > 0 else np.nan
    hit_valid = events["hit_target_first"].dropna()
    return EventStats(
        n=n,
        n_long=int((events["direction"] == 1).sum()),
        n_short=int((events["direction"] == -1).sum()),
        mean_ret_bps=float(ret.mean() * 1e4),
        median_ret_bps=float(np.median(ret) * 1e4),
        hit_rate=float((ret > 0).mean()),
        t_stat=t,
        follow_through_rate=float(hit_valid.mean()) if len(hit_valid) else np.nan,
        mean_mfe_pts=float(events["mfe"].mean()),
        mean_mae_pts=float(events["mae"].mean()),
        fade_mean_bps=float((-ret).mean() * 1e4),
    )


def stats_by_year(events: pd.DataFrame, cost_pts: float = 0.0) -> pd.DataFrame:
    """Media de retorno, hit rate y n por año (chequeo de estabilidad/decay)."""
    if len(events) == 0:
        return pd.DataFrame()
    e = events.copy()
    e["year"] = pd.to_datetime(e["session"]).dt.year
    e["ret_net"] = _cost_adjust(e["ret_to_close"].to_numpy(),
                                e["entry"].to_numpy(), cost_pts)
    out = e.groupby("year").agg(
        n=("ret_net", "size"),
        mean_bps=("ret_net", lambda x: x.mean() * 1e4),
        hit=("ret_net", lambda x: (x > 0).mean()),
    )
    out["mean_bps"] = out["mean_bps"].round(2)
    out["hit"] = out["hit"].round(3)
    return out
