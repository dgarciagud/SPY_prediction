# Breakouts intradía en SPX500 — hallazgos

**Datos:** FutureSharks / Oanda `SPX500_USD`, barras de 1 minuto, 2005–2020.
Timestamps en UTC → convertidos a `America/New_York`; RTH 09:30–16:00 ET.
3.661 sesiones (mediana 389 barras/sesión).

**Disciplina:** discovery = sesiones < 2017-01-01 (2.920 sesiones). OOS 2017+
(741 sesiones) **sellado** — no se ha tocado. Coste asumido: 1 punto de índice
round-trip (~0.5 bps al nivel del SPX). Reproducible con
`scripts/analyze_breakouts.py`.

---

## 1. El breakout incondicional no tiene edge

Un evento por sesión (primer toque), señal en cierre de barra, entrada en la
apertura de la barra siguiente, salida al cierre de la sesión.

| Setup | n | gross (bps) | t | hit | follow-through | net 1pt (bps) | t |
|-------|---|------------:|--:|----:|---------------:|--------------:|--:|
| ORB-15  | 2918 | −0.13 | −0.08 | 0.490 | 0.479 | −7.40 | −4.32 |
| ORB-30  | 2903 | +0.31 | +0.19 | 0.486 | 0.485 | −6.96 | −4.26 |
| ORB-60  | 2832 | +0.02 | +0.01 | 0.497 | 0.506 | −7.25 | −4.62 |
| Prior-day H/L | 2569 | −1.14 | −0.61 | 0.496 | — | −8.41 | −4.52 |

- Retorno medio ≈ 0 bps, hit rate ~48–49%, **t ≈ 0** en las tres ventanas ORB.
- **Follow-through** (objetivo simétrico antes que stop) ~48–50%: cara o cruz.
- MFE ≈ MAE (7.3 vs 7.6 pts): la excursión es simétrica, no hay continuación.
- Con 1 punto de coste, todo pasa a ~−7 bps (t ≈ −4.3): mecánicamente pierde.

**El SPX no "sigue" sus rupturas intradía.** Insensible a la ventana del rango
de apertura (15/30/60), lo que descarta que sea un problema de parametrización.

## 2. La única estructura condicional coherente: continuación del gap

De cinco cortes preespecificados (momento del breakout, alineación con el gap,
anchura del rango, dirección, tendencia), solo uno tiene sentido económico y
señal estadística en discovery:

| Corte (ORB-30) | n | gross (bps) | t |
|----------------|---|------------:|--:|
| dirección **==** signo(gap) | 1412 | **+6.07** | **+2.70** |
| dirección **!=** signo(gap) | 1491 | −5.15 | −2.19 |

Simétrico y coherente: los breakouts que **continúan** el gap overnight ganan;
los que lo **revierten** pierden. Es el fenómeno "gap-and-go".

## 3. …pero no sobrevive a 2008 ni a costes

| Continuación del gap | gross (bps) | t | net 1pt (bps) | t |
|----------------------|------------:|--:|--------------:|--:|
| Discovery completo | +6.07 | +2.70 | −1.17 | −0.52 |
| **Excluyendo 2008** | +3.09 | +1.56 | −4.05 | −2.04 |

- Casi la mitad del edge bruto es el año de crisis 2008 (+39 bps ese año).
- **Ex-2008 deja de ser significativo** (t=1.56) incluso en bruto.
- **Neto de coste es negativo** en ambos casos.
- 9/12 años con gross > 0, pero de magnitud pequeña y frágil.

## Veredicto

No hay edge de breakout intradía **tradeable** en este SPX a 1 minuto: ni el
incondicional, ni la mejor hipótesis condicional. El promedio plano no escondía
un tesoro; escondía 2008.

**El OOS 2017+ permanece sellado.** No hay candidato que sobreviva discovery,
así que no se gasta el OOS confirmando un negativo — esa es justo la disciplina:
el OOS se toca una sola vez, para validar un candidato real, no para pescar.

## Cómo conecta con el motor de estabilidad

Si se quisiera insistir por aquí (no recomendado con estos datos), el camino
disciplinado NO es probar más ventanas ORB hasta que una brille. Sería:

1. Registrar "continuación del gap" como **una** hipótesis en
   `hypotheses/registry.yaml` (ya anotada como *rejected* con esta evidencia).
2. Derivar sus features (tamaño del gap, hora del breakout, régimen de vol) como
   candidatas atadas a hipótesis.
3. Pasarlas por `stability_select` **una vez** — criterio: estabilidad, no Sharpe.
4. Solo si algo sobrevive con estabilidad alta, gastar el OOS una única vez.

Con lo visto, el paso 3 casi seguro no dejaría nada estable: el efecto es
demasiado dependiente de un régimen (2008) para ser reproducible.

## Limitaciones

- `SPX500_USD` es un CFD de Oanda, no el ETF SPY; el "volumen" es tick-count del
  broker, no volumen de bolsa (no se usó como feature).
- Coste modelado como fijo (1 pt); en apertura el spread real puede ser mayor.
- Salida siempre a cierre de sesión; no se exploraron gestiones de salida
  intradía alternativas (a propósito: multiplicarlas infla el presupuesto DSR).
