"""Partición temporal con OOS sellado.

El OOS protegido (por defecto 2023-01-01+) no debe tocarse durante la
búsqueda de alfa. Aquí ese principio no es una convención social: es una
invariante que el código impone.

- El objeto ``DataSplit`` expone ``discovery`` libremente.
- Acceder a ``oos`` lanza ``SealError`` salvo que se rompa el sello
  explícitamente con ``break_seal(reason)``, lo cual queda registrado.
- Se guarda un fingerprint (hash) del bloque OOS para detectar si alguien
  lo miró/alteró antes de tiempo.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


class SealError(RuntimeError):
    """Se intentó acceder al OOS sellado sin romper el sello explícitamente."""


def _fingerprint(df: pd.DataFrame) -> str:
    """Hash estable del contenido de un DataFrame (para detectar manipulación)."""
    # to_parquet/pickle no son estables entre versiones; usamos valores + índice.
    h = hashlib.sha256()
    h.update(str(df.shape).encode())
    h.update(",".join(map(str, df.columns)).encode())
    h.update(pd.util.hash_pandas_object(df.index, index=False).values.tobytes())
    h.update(pd.util.hash_pandas_object(df.reset_index(drop=True), index=False).values.tobytes())
    return h.hexdigest()


@dataclass
class DataSplit:
    """Discovery libre + OOS sellado.

    Parameters
    ----------
    frame : pd.DataFrame
        Panel completo con índice temporal (DatetimeIndex) o una columna de fecha.
    oos_start : str
        Fecha (inclusive) a partir de la cual empieza el OOS protegido.
    date_col : Optional[str]
        Nombre de la columna de fecha si el índice no es temporal.
    access_log : Optional[Path]
        Fichero donde se registra cualquier ruptura del sello.
    """

    frame: pd.DataFrame
    oos_start: str = "2023-01-01"
    date_col: Optional[str] = None
    access_log: Optional[Path] = None

    _oos_fingerprint: str = field(init=False, default="")
    _seal_broken: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        dates = self._dates()
        cutoff = pd.Timestamp(self.oos_start)
        self._disc_mask = dates < cutoff
        self._oos_mask = dates >= cutoff
        if self._oos_mask.sum() == 0:
            raise ValueError(
                f"No hay filas en el OOS (>= {self.oos_start}). "
                "Revisa oos_start o el rango de fechas."
            )
        # Fingerprint del OOS calculado UNA vez, sin exponer los datos.
        self._oos_fingerprint = _fingerprint(self.frame.loc[self._oos_mask])

    def _dates(self) -> pd.Series:
        if self.date_col is not None:
            return pd.to_datetime(self.frame[self.date_col])
        idx = self.frame.index
        if not isinstance(idx, pd.DatetimeIndex):
            raise ValueError(
                "El frame no tiene DatetimeIndex; pasa date_col con la columna de fecha."
            )
        return pd.Series(idx, index=idx)

    # -- Discovery: acceso libre --------------------------------------------
    @property
    def discovery(self) -> pd.DataFrame:
        """Bloque de discovery. Aquí es donde se hace TODA la selección."""
        return self.frame.loc[self._disc_mask].copy()

    @property
    def n_discovery(self) -> int:
        return int(self._disc_mask.sum())

    @property
    def n_oos(self) -> int:
        return int(self._oos_mask.sum())

    @property
    def oos_fingerprint(self) -> str:
        return self._oos_fingerprint

    # -- OOS: sellado --------------------------------------------------------
    @property
    def oos(self) -> pd.DataFrame:
        if not self._seal_broken:
            raise SealError(
                "El OOS está sellado. No se toca durante la búsqueda de alfa. "
                "Para el gate final (una sola vez) usa break_seal(reason=...)."
            )
        return self.frame.loc[self._oos_mask].copy()

    def break_seal(self, reason: str) -> pd.DataFrame:
        """Rompe el sello del OOS de forma explícita y auditada.

        Úsalo UNA sola vez, al final del todo, para el gate protegido.
        Cada ruptura se registra con timestamp, motivo y fingerprint.
        """
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "oos_fingerprint": self._oos_fingerprint,
            "n_oos": self.n_oos,
        }
        if self.access_log is not None:
            self.access_log.parent.mkdir(parents=True, exist_ok=True)
            with open(self.access_log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        self._seal_broken = True
        return self.frame.loc[self._oos_mask].copy()

    def summary(self) -> dict:
        return {
            "n_discovery": self.n_discovery,
            "n_oos": self.n_oos,
            "oos_start": self.oos_start,
            "oos_fingerprint": self._oos_fingerprint[:16],
            "seal_broken": self._seal_broken,
        }
