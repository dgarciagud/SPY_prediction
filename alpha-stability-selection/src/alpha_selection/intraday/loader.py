"""Carga de datos intradía de FutureSharks (Oanda SPX500_USD, 1 minuto).

Fuente: https://github.com/FutureSharks/financial-data
  pyfinancialdata/data/currencies/oanda/SPX500_USD/<year>/oanda-SPX500_USD-<year>-<m>.csv
  columnas: time, open, high, low, close, volume   (volume = tick count del broker)

Zona horaria: los timestamps están en UTC. Verificado empíricamente
(el rango intradía y la actividad culminan en 09:30-10:30 ET tras convertir
a America/New_York, y colapsan en el rollover de las 17:00 ET). Convertimos
a hora del Este para que el horario de verano (DST) se maneje solo.

RTH = Regular Trading Hours de la sesión cash de EE.UU.: 09:30-16:00 ET.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Optional

import pandas as pd

RTH_START = (9, 30)   # 09:30 ET
RTH_END = (16, 0)     # 16:00 ET (barra de las 16:00 excluida: usamos < 16:00)
TZ = "America/New_York"


def _min_of_day(ts: pd.Series) -> pd.Series:
    return ts.dt.hour * 60 + ts.dt.minute


def load_raw(spx_dir: str | Path, years: Optional[range] = None) -> pd.DataFrame:
    """Lee todos los CSV mensuales y devuelve un frame en hora del Este (tz-aware)."""
    spx_dir = Path(spx_dir)
    files = sorted(glob.glob(str(spx_dir / "*" / "*.csv")))
    if years is not None:
        keep = set(str(y) for y in years)
        files = [f for f in files if Path(f).parts[-2] in keep]
    if not files:
        raise FileNotFoundError(f"No se encontraron CSV en {spx_dir}")

    frames = []
    for f in files:
        d = pd.read_csv(f, usecols=["time", "open", "high", "low", "close", "volume"])
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize("UTC").dt.tz_convert(TZ)
    df = df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    return df


def to_rth(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra a RTH (09:30-16:00 ET), lun-vie, y añade columnas de sesión."""
    t = df["time"]
    mod = _min_of_day(t)
    start = RTH_START[0] * 60 + RTH_START[1]
    end = RTH_END[0] * 60 + RTH_END[1]
    mask = (mod >= start) & (mod < end) & (t.dt.dayofweek < 5)
    out = df.loc[mask].copy()
    out["session"] = out["time"].dt.tz_convert(TZ).dt.normalize()  # fecha de la sesión
    out["minute_from_open"] = _min_of_day(out["time"]) - start
    return out.reset_index(drop=True)


def build_rth_cache(
    spx_dir: str | Path,
    cache_path: str | Path,
    years: Optional[range] = None,
    min_bars_per_session: int = 300,
) -> pd.DataFrame:
    """Construye y cachea el panel RTH minuto a minuto.

    Descarta sesiones con muy pocas barras (festivos/medias sesiones/datos rotos):
    una sesión RTH completa tiene 390 barras; exigimos al menos ``min_bars``.
    """
    df = to_rth(load_raw(spx_dir, years))
    counts = df.groupby("session")["close"].transform("size")
    df = df.loc[counts >= min_bars_per_session].reset_index(drop=True)

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # parquet no admite tz en algunas versiones antiguas; guardamos naive-ET + flag.
    out = df.copy()
    out["time"] = out["time"].dt.tz_localize(None)
    out["session"] = out["session"].dt.tz_localize(None)
    out.to_parquet(cache_path, index=False)
    return df


def load_rth_cache(cache_path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(cache_path)
    df["time"] = pd.to_datetime(df["time"])
    df["session"] = pd.to_datetime(df["session"])
    return df
