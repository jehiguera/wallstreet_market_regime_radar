# Wall Street Regime Radar

Live app: https://wallstreetregimeradar.streamlit.app/


Intraday + weekly market traffic light with cross-asset comparisons and risk-on score
<img width="1723" height="831" alt="image" src="https://github.com/user-attachments/assets/86d514f9-af68-4b21-89b7-db59d3a78925" />


# What this app does

A lightweight Streamlit dashboard that detects the current market regime (Green = uptrend, Yellow = sideways, Red = downtrend) for SPY/S&P 500, and complements it with:

Intraday “Traffic Light” (5–15m bars, 09:30–16:00 ET).

Weekly panel (1h bars, last 2 calendar weeks; axis labels show Mon–Fri only).

Cross-asset view (daily): SPX/BTC/Gold/Oil

Rolling correlation vs SP (30d default) with quick labels:
+1 = tranquil, 0 = diversification, -1 = hedge.

Rolling beta vs SP (30d default) with quick labels:
< 1 = defensive, = 1 = market-like, > 1 = expansionary.

30-day dominance map (who’s leading: SP / BTC / GOLD / OIL).

Risk-On score (simple macro pulse):
z(SP_30d) + z(BTC_30d) – z(GOLD_30d) – z(OIL_30d)
Smoothed with EMA; thresholds: > +0.5 = RISK-ON, < −0.5 = RISK-OFF, else Neutral.

# How the regime is computed

Trend filter = ADX >= threshold AND |slope(log price)| >= percentile.

Coloring:

GREEN if slope > 0 under trend filter.

RED if slope < 0 under trend filter.

YELLOW otherwise (range).

Smoothing: short flips shorter than MIN_RUN_BARS are neutralized to YELLOW.

Final daily label: prefers the last run if it’s strong enough; otherwise the dominant color (time share).

# Author
Jorge Higuera Portilla
Email: jhiguera@ieee.org
# License
This project is licensed under the MIT License. See the LICENSE file for more details.


