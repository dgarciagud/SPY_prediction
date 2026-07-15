#!/usr/bin/env python
"""Corre el procedimiento de selección por estabilidad UNA vez y lo registra.

Uso:
    python scripts/run_selection.py \
        --data path/al/discovery.parquet \
        --registry hypotheses/registry.yaml \
        --config config/selection.yaml \
        --label baseline

Contrato:
  - Lee SOLO discovery (el OOS queda sellado por DataSplit).
  - Aplica el gate de hipótesis: solo entran candidatas activas.
  - Corre stability_select una vez.
  - Registra el resultado como UNA variante en logs/variants.jsonl.
  - Imprime el reporte. No rankea subconjuntos. No mira Sharpe.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alpha_selection.registry import HypothesisRegistry            # noqa: E402
from alpha_selection.splits import DataSplit                        # noqa: E402
from alpha_selection.stability import StabilityConfig, stability_select  # noqa: E402
from alpha_selection.variants import log_variant                   # noqa: E402
from alpha_selection.report import render_report                   # noqa: E402
from alpha_selection.splits import _fingerprint                    # noqa: E402


def load_config(path: Path) -> StabilityConfig:
    with open(path, encoding="utf-8") as fh:
        d = yaml.safe_load(fh)
    d["l1_ratios"] = tuple(d.get("l1_ratios", (0.3, 0.5, 0.7, 0.9)))
    return StabilityConfig(**d)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="Parquet/CSV con fecha + features + target")
    ap.add_argument("--registry", default=str(ROOT / "hypotheses/registry.yaml"))
    ap.add_argument("--config", default=str(ROOT / "config/selection.yaml"))
    ap.add_argument("--log", default=str(ROOT / "logs/variants.jsonl"))
    ap.add_argument("--oos-access-log", default=str(ROOT / "logs/oos_access.jsonl"))
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--label", default="")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    reg = HypothesisRegistry.load(args.registry)

    path = Path(args.data)
    if path.suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    reg.validate_against_columns(list(df.columns))

    # Sella el OOS. A partir de aquí solo tocamos discovery.
    split = DataSplit(
        df,
        oos_start=reg.oos_start,
        date_col=args.date_col,
        access_log=Path(args.oos_access_log),
    )
    disc = split.discovery

    print(reg.gate_report())
    print(f"OOS sellado       : {split.n_oos} filas  (fingerprint {split.oos_fingerprint[:12]})")
    print()

    X = disc[reg.active_features].astype(float)
    y = disc[reg.target_name]

    cfg = load_config(Path(args.config))
    cfg.task = reg.task  # el registro manda sobre la tarea

    result = stability_select(X, y, cfg)
    print(render_report(result))

    data_fp = _fingerprint(disc[reg.active_features + [reg.target_name]])
    record = log_variant(
        result,
        log_path=Path(args.log),
        data_fingerprint=data_fp,
        label=args.label,
        oos_fingerprint=split.oos_fingerprint,
        notes=args.notes,
    )
    print()
    print(f"Variante #{record['variant_index']} registrada en {args.log}")
    print(f"  presupuesto DSR de selección hasta ahora: {record['variant_index']} variante(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
