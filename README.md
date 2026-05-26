# SPY_prediction
# EBM Sector ETF Triple Barrier Pipeline

Pipeline cuantitativo para:

- Descargar OHLCV diario gratuito de ETFs sectoriales y SPY.
- Construir features de retorno, momentum, macro FRED y neutralización cross-sectional.
- Etiquetar con triple barrera usando volatilidad rolling de SPY.
- Entrenar un Explainable Boosting Machine Classifier (EBM).
- Hyperparametros del EBM definidos inicialmente por heuristica.
- Optimizar con purged walk-forward expanding window.
- Sacar probabilidad de backtest overfitting de Marcos Lopez de Prado.
- Realizar cluster permutation importance y sacar gráfico.
- Reservar 2022-2026 como periodo out-of-sample intocable.

## Activos

Sector ETFs:

- XLE
- XLB
- XLI
- XLK
- XLF
- XLP
- XLY
- XLV
- XLU

Benchmark:

- SPY

## Target

Triple barrera:

- TP = 1 * desviación estándar rolling de 20 días de SPY
- SL = 1 * desviación estándar rolling de 20 días de SPY
- Vertical barrier = 7 sesiones

## Validación

- Walk-forward con ventana expanding.
- Validación de 3 meses.
- Purging usando `event_end < validation_start`.
- OOS intocable desde 2022-01-01.

## Métrica objetivo

Average Precision Lift:

```text
AP Lift = Average Precision / frecuencia de la clase positiva
