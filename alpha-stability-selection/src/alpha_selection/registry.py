"""Carga y validación del registro de hipótesis.

El gate: toda característica candidata que entra al procedimiento de selección
debe colgar de una hipótesis con ``status: active``. Sin hipótesis, no entra.
Esto convierte "no tengo hipótesis" en un error explícito, no en una excusa.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import yaml


@dataclass
class HypothesisRegistry:
    raw: dict
    target_name: str
    task: str
    oos_start: str
    active_features: List[str]
    feature_to_hypothesis: Dict[str, str]
    parked: List[str]

    @classmethod
    def load(cls, path: str | Path) -> "HypothesisRegistry":
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        target = raw.get("target", {})
        meta = raw.get("meta", {})
        feats: List[str] = []
        f2h: Dict[str, str] = {}
        for hyp in raw.get("hypotheses", []):
            if hyp.get("status") != "active":
                continue
            hid = hyp["id"]
            for f in hyp.get("features", []):
                if f in f2h:
                    raise ValueError(
                        f"Característica '{f}' asignada a dos hipótesis: "
                        f"'{f2h[f]}' y '{hid}'. Cada candidata cuelga de UNA hipótesis."
                    )
                f2h[f] = hid
                feats.append(f)

        parked = [p["name"] for p in raw.get("parked", [])]

        return cls(
            raw=raw,
            target_name=target.get("name", "target"),
            task=target.get("task", "classification"),
            oos_start=meta.get("oos_start", "2023-01-01"),
            active_features=feats,
            feature_to_hypothesis=f2h,
            parked=parked,
        )

    def validate_against_columns(self, columns: List[str]) -> None:
        """Comprueba que todas las candidatas activas existen en los datos.

        No exige lo contrario: puede haber columnas en los datos que no sean
        candidatas (fechas, target, auxiliares). Pero toda candidata activa
        debe existir, o el registro miente sobre lo que hay.
        """
        missing = [f for f in self.active_features if f not in columns]
        if missing:
            raise ValueError(
                "Candidatas activas ausentes en los datos: "
                f"{missing}. Corrige el registro o el cálculo de features."
            )

    def gate_report(self) -> str:
        n_hyp = len([h for h in self.raw.get("hypotheses", []) if h.get("status") == "active"])
        lines = [
            f"Hipótesis activas : {n_hyp}",
            f"Candidatas (gate) : {len(self.active_features)}",
            f"En cuarentena     : {len(self.parked)} (fuera del pool)",
            f"Target / tarea    : {self.target_name} / {self.task}",
        ]
        return "\n".join(lines)
