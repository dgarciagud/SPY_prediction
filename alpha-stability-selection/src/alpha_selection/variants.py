"""Registro de variantes — contabilidad del presupuesto DSR.

Cada corrida del procedimiento de selección es UNA variante. Este módulo la
registra de forma append-only e inmutable-en-espíritu: el log es la verdad
sobre cuántas variantes has probado, que es justo lo que el Deflated Sharpe
Ratio necesita para no autoengañarte.

Un run honesto = una línea aquí. Si acabas con 20 líneas "de selección",
tu presupuesto DSR es 20, no 1 — y el log te lo recuerda.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .stability import StabilityResult


def _config_hash(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class VariantLog:
    """Log append-only de variantes de selección (JSONL)."""

    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def count(self) -> int:
        if not self.path.exists():
            return 0
        with open(self.path, encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())

    def append(self, record: dict) -> str:
        record = dict(record)
        record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        record.setdefault("variant_index", self.count() + 1)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        return record["timestamp"]

    def all(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    out.append(json.loads(line))
        return out


def log_variant(
    result: StabilityResult,
    log_path: Path,
    data_fingerprint: str,
    label: str = "",
    oos_fingerprint: Optional[str] = None,
    notes: str = "",
) -> dict:
    """Registra el resultado de UN procedimiento de selección como una variante.

    Parameters
    ----------
    result : StabilityResult
        Salida de ``stability_select``.
    log_path : Path
        Fichero JSONL del log de variantes.
    data_fingerprint : str
        Hash del bloque de discovery usado (para reproducibilidad).
    label : str
        Etiqueta legible ("baseline", "sin macro", ...).
    oos_fingerprint : Optional[str]
        Hash del OOS sellado, para probar que NO cambió entre variantes.
    notes : str
        Notas libres.
    """
    payload = {
        "config": result.config.as_dict(),
        "data_fingerprint": data_fingerprint,
        "n_candidates": len(result.features),
    }
    record = {
        "label": label,
        "config_hash": _config_hash(payload),
        "data_fingerprint": data_fingerprint,
        "oos_fingerprint": oos_fingerprint,
        "n_samples": result.n_samples,
        "n_candidates": len(result.features),
        "stability_threshold": result.config.stability_threshold,
        "aggregator": result.config.aggregator,
        "stable_set": result.stable_set,
        "final_set": result.final_set,
        "trim_applied": result.trim_applied,
        "dropped_for_parsimony": result.dropped_for_parsimony,
        "freq_combined": {k: round(v, 4) for k, v in result.freq_combined.items()},
        "notes": notes,
    }
    log = VariantLog(log_path)
    ts = log.append(record)
    record["timestamp"] = ts
    record["variant_index"] = log.count()
    return record
