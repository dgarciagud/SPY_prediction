"""alpha_selection — selección de características por estabilidad, un solo procedimiento.

Filosofía (ver README):
  - La selección se hace SOLO sobre discovery, con estabilidad de selección
    como criterio (¿la variable sobrevive en el 80%+ de los bootstraps?),
    NUNCA con Sharpe/rendimiento OOS como criterio.
  - Un run = una variante en el log DSR. No hay torneo de subconjuntos.
  - El OOS (2023+) está sellado y no se toca hasta el gate final.
"""

from .splits import DataSplit, SealError
from .stability import (
    StabilityConfig,
    StabilityResult,
    stability_select,
    elasticnet_selection_frequency,
    shallow_tree_selection_frequency,
)
from .variants import VariantLog, log_variant

__all__ = [
    "DataSplit",
    "SealError",
    "StabilityConfig",
    "StabilityResult",
    "stability_select",
    "elasticnet_selection_frequency",
    "shallow_tree_selection_frequency",
    "VariantLog",
    "log_variant",
]

__version__ = "0.1.0"
