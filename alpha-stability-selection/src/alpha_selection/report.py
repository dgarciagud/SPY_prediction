"""Reporte legible del resultado de selección por estabilidad."""

from __future__ import annotations

from .stability import StabilityResult


def render_report(result: StabilityResult) -> str:
    cfg = result.config
    lines: list[str] = []
    a = lines.append

    a("=" * 68)
    a("SELECCIÓN POR ESTABILIDAD  —  una variante")
    a("=" * 68)
    a(f"Tarea               : {cfg.task}")
    a(f"Muestras (discovery): {result.n_samples}")
    a(f"Candidatas          : {len(result.features)}")
    a(f"Bootstraps          : {cfg.n_bootstraps}")
    a(f"Umbral estabilidad π: {cfg.stability_threshold:.0%}")
    a(f"Agregador           : {cfg.aggregator}")
    a(f"Tope parsimonia     : {cfg.max_features}")
    a("")

    a("Frecuencia de selección por característica")
    a("-" * 68)
    a(f"{'característica':<28}{'elasticnet':>12}{'árbol':>10}{'combinada':>12}  ")
    a("-" * 68)
    order = sorted(result.features, key=lambda f: result.freq_combined[f], reverse=True)
    pi = cfg.stability_threshold
    for f in order:
        en = result.freq_elasticnet[f]
        tr = result.freq_tree[f]
        cb = result.freq_combined[f]
        mark = "  <= estable" if f in result.stable_set else ""
        star = " *" if f in result.final_set else "  "
        a(f"{f:<28}{en:>11.2f} {tr:>9.2f} {cb:>11.2f}{star}{mark}")
    a("-" * 68)
    a("(*) = en el conjunto final tras recorte por parsimonia")
    a("")

    a(f"Conjunto estable ( >= π ) : {result.stable_set}")
    if result.trim_applied:
        a(f"Recortadas por parsimonia : {result.dropped_for_parsimony}")
    a(f"CONJUNTO FINAL            : {result.final_set}")
    a("")

    n_stable = len(result.stable_set)
    if n_stable == 0:
        a("AVISO: ninguna característica supera el umbral de estabilidad.")
        a("       No hay estructura reproducible aquí. No fuerces una selección:")
        a("       revisa las hipótesis o baja expectativas, no el umbral.")
    elif n_stable > cfg.max_features:
        a(f"NOTA: {n_stable} superaron el umbral; se recortó a {cfg.max_features}")
        a("      por parsimonia (reproducibilidad + de-correlación), NO por Sharpe.")
    a("=" * 68)
    return "\n".join(lines)
