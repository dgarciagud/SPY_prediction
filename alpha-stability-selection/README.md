# alpha-stability-selection

Búsqueda de alfa disciplinada por DSR. La selección de características es **un
solo procedimiento, no un torneo**: se corre una vez sobre *discovery* y el
criterio es la **estabilidad de selección**, nunca el Sharpe de validación.

## El argumento (por qué existe esto)

- Si no puedes atar cada candidata a una **hipótesis económica**, no tienes
  hipótesis: cualquier reducción de 20→4 sería descubrimiento dirigido por
  datos, justo lo que infla el presupuesto DSR.
- La salida legítima no es "probar 5 subconjuntos y quedarme con el mejor OOS".
  Es un método de selección de **una sola variante**: ElasticNet o bootstrap de
  árboles poco profundos sobre discovery, corrido una vez.
- Miras qué características **sobreviven con alta estabilidad** (¿aparecen en el
  80%+ de los bootstraps?) y descartas el resto. No comparas rendimiento de
  subconjuntos: inspeccionas estructura.
- El momento en que rankeas subconjuntos por su rendimiento OOS, sobreajustaste.

El orden real que implementa este repo:

1. Corre ElasticNet + bootstrap de estabilidad **una vez** sobre discovery →
   se registra como **una** variante (`logs/variants.jsonl`).
2. Te quedas con las que superan el **umbral de estabilidad**, no las de mejor
   coeficiente.
3. Si sobreviven más de 5, recorta por **parsimonia** (reproducibilidad +
   de-correlación), no por rendimiento.
4. El **OOS protegido (2023+)** está sellado y no se toca hasta el gate final,
   y solo una vez.

> ~890 labels efectivos para ~20 candidatas es una señal/ruido pésima. Por eso
> el bootstrap de estabilidad importa más que la magnitud del coeficiente: te
> dice si la selección es **reproducible o accidental**.

## Estructura

```
hypotheses/registry.yaml     Hipótesis económicas -> candidatas (el gate)
config/selection.yaml        El procedimiento (= una variante). No es un grid.
src/alpha_selection/
  splits.py                  OOS sellado (SealError si lo tocas antes de tiempo)
  registry.py                Carga/valida el gate de hipótesis
  stability.py               Selección por estabilidad: ElasticNet + árboles
  variants.py                Log append-only = contabilidad del presupuesto DSR
  report.py                  Reporte legible
scripts/
  run_selection.py           Corre 1 procedimiento y lo registra como 1 variante
  make_synthetic.py          Panel sintético para smoke-test
logs/variants.jsonl          Cuántas variantes has probado (versionado)
tests/                       Recupera señal, descarta ruido, sella OOS
```

## Uso

```bash
pip install -r requirements.txt

# 1) smoke-test con datos sintéticos de estructura conocida
python scripts/make_synthetic.py --out data/synthetic_discovery.parquet
python scripts/run_selection.py --data data/synthetic_discovery.parquet --label smoke

# 2) con tus datos reales (date + candidatas del registro + target)
python scripts/run_selection.py \
    --data data/discovery.parquet \
    --registry hypotheses/registry.yaml \
    --config config/selection.yaml \
    --label baseline
```

`run_selection.py` sella el OOS, aplica el gate de hipótesis, corre el
procedimiento una vez, imprime el reporte y **añade una línea** al log de
variantes. Si vuelves a correrlo con otra config, eso es **otra variante**: el
log te lo cuenta para que tu DSR use el número honesto.

## Lo que este repo NO hace (a propósito)

- No rankea subconjuntos por Sharpe.
- No baja el umbral de estabilidad para "sacar" características.
- No toca el OOS. Romper el sello exige `break_seal(reason=...)` y queda
  auditado en `logs/oos_access.jsonl`.

Si nada supera el umbral, la respuesta correcta no es aflojar el umbral: es que
no hay estructura reproducible y toca revisar las hipótesis.

## Estudio intradía: breakouts en SPX500

`src/alpha_selection/intraday/` aplica la misma disciplina a datos de 1 minuto
de FutureSharks (Oanda `SPX500_USD`, 2005–2020): discovery vs OOS 2017+ sellado,
coste explícito, sin torneo de parámetros.

```bash
# reconstruir caché desde los CSV de FutureSharks y analizar
python scripts/analyze_breakouts.py --spx-dir <.../SPX500_USD> --cache data/spx_rth.parquet --build
```

**Resultado (ver `src/alpha_selection/intraday/FINDINGS.md`):** el breakout
intradía no tiene edge tradeable en el SPX. El incondicional da t≈0; la única
hipótesis condicional coherente (continuación del gap) da t=2.70 en bruto pero
se cae ex-2008 y es negativa neta de coste. El OOS quedó **sellado** — no hay
candidato que validar.

## Tests

```bash
pytest -q
```
