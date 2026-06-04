# SPY_prediction — EBM Triple Barrier Pipeline (Long + Short)

Pipeline cuantitativo con **dos modelos independientes** (long y short side) basados en Explainable Boosting Machines (EBM), triple-barrier labeling y meta-labeling de López de Prado.

## Notebooks

- **`ebm_sector_etf_pipeline.ipynb`** — modelo LONG (target = TP hit del SPY)
- **`ebm_sector_etf_pipeline_short.ipynb`** — modelo SHORT (target = SL hit del SPY)

Ambos comparten arquitectura: features per-ticker (SPY + 9 sectoriales) + cross-sectional summary + macro FRED → primary EBM → meta-labeling → P&L OOS → live signal.

## Pipeline (por notebook)

1. **Datos** — OHLCV diario gratuito (yfinance) + macro FRED (pandas-datareader)
2. **Features** — retornos (5/20/60d), momentum (63/252d), volatilidad 20d, RS vs SPY, cross-sectional mean/std
3. **Triple Barrier** — TP/SL = 1× σ20d, vertical = 7 sesiones, **entry = open del día siguiente a la señal**
4. **Uniqueness (López de Prado)** — `sample_weight` por evento = avgUniqueness
5. **EBM Primary** — walk-forward purgado, AP-Lift como métrica
6. **Threshold calibration** — WF + CSCV PBO
7. **Meta-labeling** — filtro secundario de falsos positivos
8. **P&L OOS** — Primary, Primary+Meta y SPY Buy&Hold comparados (Sharpe, CAGR, MaxDD)
9. **PBO (CSCV)** — sobre selección de configs meta y primary
10. **Cluster Permutation Importance**
11. **Production Fit** — modelos entrenados sobre IS+OOS completo (deployment)
12. **Live Signal** — decisión accionable de hoy + TP/SL en puntos

## Activos

- **Sectoriales**: XLE, XLB, XLI, XLK, XLF, XLP, XLY, XLV, XLU
- **Benchmark / Target**: SPY

## OOS

Desde `2022-01-01`. Reentrenamiento periódico no toca este período en research; el `final_prod_ebm` sí lo incluye para deployment.

## Convención de ejecución (P&L)

- **Señal**: cierre del día `i`
- **Entrada**: **OPEN del día `i+1`** (next business day)
- **TP long**: `entry × (1 + σ20d)` | **TP short**: `entry × (1 − σ20d)`
- **SL long**: `entry × (1 − σ20d)` | **SL short**: `entry × (1 + σ20d)`
- **Salida**: TP/SL/vertical (máx 7 sesiones)

## CI — GitHub Actions

El workflow `.github/workflows/run_notebook.yml` ejecuta ambos notebooks (long y short) en paralelo en cada push a `main`. Inputs manuales (`workflow_dispatch`):
- `fast_mode`: `true`/`false`
- `run_side`: `both`/`long`/`short`

Artifacts: notebook ejecutado + HTML + plots, retenidos 30 días.
