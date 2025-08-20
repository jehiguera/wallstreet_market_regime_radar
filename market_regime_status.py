
# -*- coding: utf-8 -*-
"""
Wall Street Market Regime analyzer (Week and Intraday) 
author: Jorge Higuera
email: jhiguera@ieee.org
date: 20/08/2025
"""
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter, NullLocator, NullFormatter
import yfinance as yf
import streamlit as st

# =======================
# Config UI (sidebar)
# =======================
st.set_page_config(page_title="Semaforo de mercado (intradia + cross-asset)", layout="wide")
st.title("Semaforo de mercado: intradia + semanal + cross-asset")

with st.sidebar:
    st.header("Parametros")
    RUN_CROSS_ASSET = st.checkbox("Comparativa cross-asset + Risk-On", value=True)
    TICKER_PRI = st.selectbox("Ticker principal", ["SPY", "^GSPC"], index=0)
    PERIOD_INTRADAY   = st.selectbox("Periodo intradia", ["5d", "10d"], index=0)
    INTERVAL_INTRADAY = st.selectbox("Intervalo intradia", ["5m", "15m"], index=0)
    PERIOD_WEEKLY_STRIP   = st.selectbox("Periodo semanal (para 1h)", ["30d", "45d"], index=0)
    INTERVAL_WEEKLY_STRIP = "60m"
    WEEKLY_WINDOW_DAYS    = st.slider("Ventana panel semanal (dias)", 7, 21, 14)

    st.divider()
    st.caption("Reglas de tendencia")
    ADX_WINDOW    = st.slider("ADX_WINDOW", 8, 28, 14)
    SLOPE_WINDOW  = st.slider("SLOPE_WINDOW", 6, 24, 12)
    SLOPE_PCTL    = st.slider("SLOPE_PCTL (percentil |slope|)", 40, 80, 60)
    ADX_TREND_MIN = st.slider("ADX_TREND_MIN", 10, 30, 18)

    st.caption("Suavizado/decision")
    USE_SMOOTH            = st.checkbox("Suavizar regimen (MIN_RUN_BARS)", value=True)
    MIN_RUN_BARS          = st.slider("MIN_RUN_BARS", 1, 6, 3)
    LAST_MIN_RUN_CONFIRM  = st.slider("LAST_MIN_RUN_CONFIRM", 1, 8, 4)
    DOMINANCE_MIN_PCT     = st.slider("DOMINANCE_MIN_PCT (%)", 30, 70, 45)

    st.divider()
    st.caption("Cross-asset y Risk-On")
    PERIOD_DAILY   = "180d"
    INTERVAL_DAILY = "1d"
    SP_BENCH = st.selectbox("Benchmark SP", ["^GSPC","SPY"], index=0)
    ROLL_CORR_WIN = st.slider("Ventana corr (dias)", 20, 90, 30)
    ROLL_BETA_WIN = st.slider("Ventana beta (dias)", 20, 90, 30)
    DOM_RET_WIN   = st.slider("Ventana dominancia (dias)", 20, 90, 30)
    RISK_Z_WIN    = st.slider("Ventana Risk-On z-score (dias)", 60, 180, 90)
    RISK_EMA_SPAN = st.slider("EMA suavizado Risk-On", 5, 30, 10)

    st.divider()
    refresh = st.button("Refrescar datos")

if refresh:
    # simple “throttle” para evitar múltiples clics
    time.sleep(0.1)

# =======================
# Utilidades/indicadores 
# =======================
CANON = {
    "open": ["open", "Open", "OPEN"],
    "high": ["high", "High", "HIGH"],
    "low":  ["low", "Low", "LOW"],
    "close": ["close", "Close", "CLOSE"],
    "adj_close": ["adj close", "Adj Close", "AdjClose", "adj_close"],
    "volume": ["volume", "Volume", "VOLUME"],
}

def _to_series(obj) -> pd.Series:
    if isinstance(obj, pd.DataFrame):
        return pd.to_numeric(obj.iloc[:, 0], errors="coerce")
    return pd.to_numeric(obj, errors="coerce")

def true_range(h, lo, c):
    h = _to_series(h); lo = _to_series(lo); c = _to_series(c)
    pc = c.shift(1)
    return pd.concat([(h - lo).abs(), (h - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)

def adx(high, low, close, n=14):
    h  = _to_series(high)
    lo = _to_series(low)
    c  = _to_series(close)
    up = h.diff()
    dn = -lo.diff()
    plus_dm  = np.where((up > dn) & (up > 0),  up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0),  dn, 0.0)
    tr = true_range(h, lo, c)
    atr = pd.Series(tr, index=h.index).ewm(alpha=1/n, adjust=False).mean()
    plus_di  = 100 * pd.Series(plus_dm,  index=h.index).ewm(alpha=1/n, adjust=False).mean() / (atr + 1e-12)
    minus_di = 100 * pd.Series(minus_dm, index=h.index).ewm(alpha=1/n, adjust=False).mean() / (atr + 1e-12)
    dx = 100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-12)
    adx_val = dx.ewm(alpha=1/n, adjust=False).mean()
    return adx_val, plus_di, minus_di

def rolling_slope_logprice(close: pd.Series, window: int) -> pd.Series:
    y = np.log(close.clip(lower=1e-8))
    x = np.arange(window)
    def slope_func(a):
        xm = x.mean(); ym = a.mean()
        num = ((x - xm) * (a - ym)).sum()
        den = ((x - xm)**2).sum()
        return num / den if den != 0 else 0.0
    return y.rolling(window, min_periods=window).apply(slope_func, raw=True)

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [" ".join([str(x) for x in tup if x is not None]).strip()
                      for tup in df.columns.to_flat_index()]
    return df

def _find_by_alias(lower_map: dict, aliases: list) -> str | None:
    for alias in aliases:
        al = alias.lower()
        if al in lower_map: return lower_map[al]
    for alias in aliases:
        al = alias.lower()
        for k_lower, orig in lower_map.items():
            if k_lower.startswith(al + " ") or k_lower == al:
                return orig
    for alias in aliases:
        al = alias.lower()
        for k_lower, orig in lower_map.items():
            if al in k_lower:
                return orig
    return None

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = _flatten_columns(df.copy())
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    out = pd.DataFrame(index=df.index)
    for key, aliases in CANON.items():
        col = _find_by_alias(lower_map, aliases)
        if col is not None:
            out[key] = _to_series(df[col])
    if "close" not in out.columns:
        alt = _find_by_alias(lower_map, CANON["adj_close"])
        if alt is not None:
            out["close"] = _to_series(df[alt])
    if "adj_close" not in out.columns and "close" in out.columns:
        out["adj_close"] = out["close"]
    required = ["open", "high", "low", "close", "volume"]
    missing = [k for k in required if k not in out.columns]
    if missing:
        raise ValueError(f"Faltan columnas OHLCV tras normalizar: {missing}. Columnas originales: {list(df.columns)}")
    return out

def _to_us_eastern_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    try:
        if idx.tz is not None:
            return idx.tz_convert("America/New_York").tz_localize(None)
        else:
            return idx
    except Exception:
        return idx

def _filter_business_hours(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out = out[out.index.weekday < 5]
    if isinstance(interval, str) and (interval.endswith("m") or interval.endswith("h")):
        try:
            out = out.between_time("09:30", "16:00")
        except TypeError:
            out = out.between_time("09:30", "16:00")
    return out

# Cache download 5 min 
@st.cache_data(ttl=300, show_spinner=False)
def _yf_download(ticker: str, period: str, interval: str):
    return yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False, prepost=False)

def load_any(tickers, period, interval):
    last_err = None
    for tk in tickers:
        try:
            raw = _yf_download(tk, period, interval)
            if raw is None or raw.empty: continue
            df = _normalize_ohlcv(raw)
            df.index = pd.to_datetime(df.index)
            df.index = _to_us_eastern_index(df.index)
            df = _filter_business_hours(df, interval)
            return tk, df.dropna()
        except Exception as e:
            last_err = e
            continue
    st.warning(f"No se pudieron normalizar datos. Ultimo error: {last_err}")
    return None, pd.DataFrame()

def _smooth_regime(reg_series: pd.Series, min_run: int = 3) -> pd.Series:
    if not USE_SMOOTH: return reg_series
    arr = reg_series.to_numpy(copy=True)
    if len(arr) == 0: return reg_series
    run = 1
    for i in range(1, len(arr)):
        if arr[i] == arr[i-1]:
            run += 1
        else:
            if run < min_run:
                arr[i-run:i] = "YELLOW"
            run = 1
    if run < min_run:
        arr[len(arr)-run:] = "YELLOW"
    return pd.Series(arr, index=reg_series.index)

def _last_run_length(series: pd.Series) -> int:
    if series.empty: return 0
    last = series.iloc[-1]; run = 1
    for i in range(len(series)-2, -1, -1):
        if series.iloc[i] == last: run += 1
        else: break
    return run

def compute_regime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret"] = out["close"].pct_change()
    out["ADX"], out["+DI"], out["-DI"] = adx(out["high"], out["low"], out["close"], ADX_WINDOW)
    out["slope"] = rolling_slope_logprice(out["close"], SLOPE_WINDOW)
    slope_thr = np.nanpercentile(out["slope"].abs().dropna(), SLOPE_PCTL) if out["slope"].notna().sum() > 10 else 0.0
    out["regime"] = "YELLOW"
    mask_trend = (out["ADX"] >= ADX_TREND_MIN) & (out["slope"].abs() >= slope_thr)
    out.loc[mask_trend & (out["slope"] > 0), "regime"] = "GREEN"
    out.loc[mask_trend & (out["slope"] < 0), "regime"] = "RED"
    out["regime_smooth"] = _smooth_regime(out["regime"], MIN_RUN_BARS)
    out.attrs["slope_thr"] = float(slope_thr)
    return out

def summarize_today_with_combined_rule(df: pd.DataFrame):
    today = df.index[-1].date()
    day_df = df[df.index.date == today].copy()
    if day_df.empty:
        last_day = df.index[-1].date()
        day_df = df[df.index.date == last_day].copy()
        today = last_day
    reg_col = "regime_smooth" if "regime_smooth" in day_df.columns else "regime"
    ret_today = (day_df["close"].iloc[-1] / day_df["open"].iloc[0] - 1.0) * 100.0
    vol_today = day_df["ret"].std() * np.sqrt(len(day_df)) * 100.0 if day_df["ret"].notna().sum() > 2 else np.nan
    counts = day_df[reg_col].value_counts(dropna=False); total = len(day_df)
    pct_green  = 100.0 * counts.get("GREEN", 0)  / total if total else 0.0
    pct_yellow = 100.0 * counts.get("YELLOW", 0) / total if total else 0.0
    pct_red    = 100.0 * counts.get("RED", 0)    / total if total else 0.0
    last_regime = day_df[reg_col].iloc[-1]
    last_run = _last_run_length(day_df[reg_col])
    dominant_regime = counts.idxmax() if total else "YELLOW"
    if (last_run >= LAST_MIN_RUN_CONFIRM) or \
       ((last_regime == "GREEN" and pct_green >= DOMINANCE_MIN_PCT) or
        (last_regime == "RED"   and pct_red   >= DOMINANCE_MIN_PCT) or
        (last_regime == "YELLOW"and pct_yellow>= DOMINANCE_MIN_PCT)):
        final_regime = last_regime
        reason = f"prefer last (run={last_run} >= {LAST_MIN_RUN_CONFIRM} or share >= {DOMINANCE_MIN_PCT}%)"
    else:
        final_regime = dominant_regime
        reason = "prefer dominant (insufficient last run/share)"
    return {
        "date": today,
        "ret_today_pct": ret_today,
        "vol_today_pct": vol_today,
        "pct_green": pct_green,
        "pct_yellow": pct_yellow,
        "pct_red": pct_red,
        "last_regime": last_regime,
        "last_run": last_run,
        "dominant_regime": dominant_regime,
        "final_regime": final_regime,
        "decision_reason": reason
    }, day_df

# =======================
# Plot helpers
# =======================
def _dates_business_days(ax):
    wd = mdates.WeekdayLocator(byweekday=(mdates.MO, mdates.TU, mdates.WE, mdates.TH, mdates.FR))
    ax.xaxis.set_major_locator(wd)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %m-%d'))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())

def _dates_concise(ax):
    loc = mdates.AutoDateLocator(minticks=4, maxticks=7)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

def _color_of(reg):
    return {"GREEN":"#2ecc71", "RED":"#e74c3c"}.get(reg, "#f1c40f")

def make_fig_semaforo(day_df: pd.DataFrame, weekly_df: pd.DataFrame|None,
                      ticker: str, slope_thr: float, weekly_days: int):
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(13.5, 7.2), constrained_layout=False)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    fig.subplots_adjust(hspace=0.33)

    # Panel superior
    ax_top = fig.add_subplot(gs[0])
    if weekly_df is not None and not weekly_df.empty:
        reg_col = "regime_smooth" if "regime_smooth" in weekly_df.columns else "regime"
        end = weekly_df.index.max()
        start = end - pd.Timedelta(days=weekly_days)
        wdf = weekly_df.loc[(weekly_df.index >= start) & (weekly_df.index <= end)].copy()
        ax_top.plot(wdf.index, wdf["close"], color="#34495e", linewidth=1.5, label=f"{ticker} Close (1h)")
        start_blk = None; prev_reg = None
        for t, reg in wdf[reg_col].items():
            if prev_reg is None:
                prev_reg = reg; start_blk = t
            elif reg != prev_reg:
                ax_top.axvspan(start_blk, t, color=_color_of(prev_reg), alpha=0.15)
                prev_reg = reg; start_blk = t
        if start_blk is not None:
            ax_top.axvspan(start_blk, wdf.index[-1], color=_color_of(prev_reg), alpha=0.15)
        from matplotlib.patches import Patch
        legend_patches = [
            Patch(facecolor="#2ecc71", alpha=0.15, label="GREEN tendencia alcista"),
            Patch(facecolor="#e74c3c", alpha=0.15, label="RED tendencia bajista"),
            Patch(facecolor="#f1c40f", alpha=0.15, label="YELLOW lateral"),
        ]
        ax_top.legend(handles=legend_patches, loc="upper left")
        thr_w = weekly_df.attrs.get("slope_thr", slope_thr)
        ax_top.set_title(f"Wall street market week market regime status 1h (last {weekly_days} days, smoot) - ADX>={ADX_TREND_MIN}, |slope|>=p{SLOPE_PCTL}, umbral slope: {thr_w:.2e}", pad=8)
        ax_top.set_ylabel("Price (USD)")
        ax_top.grid(True, linestyle="--", alpha=0.25)
        ax_top.set_xlim(wdf.index.min(), wdf.index.max())
        _dates_business_days(ax_top)
        ax_top.tick_params(axis="x", labelbottom=True, labelsize=9)
    else:
        ax_top.text(0.5, 0.5, "not available data", ha="center", va="center")
        ax_top.set_yticks([])

    # Panel inferior (hoy)
    ax = fig.add_subplot(gs[1])
    ax.plot(day_df.index, day_df["close"], color="#34495e", linewidth=1.5, label=f"{ticker} Close")
    reg_col2 = "regime_smooth" if "regime_smooth" in day_df.columns else "regime"
    start = None; prev_reg = None
    for t, reg in day_df[reg_col2].items():
        if prev_reg is None:
            prev_reg = reg; start = t
        elif reg != prev_reg:
            ax.axvspan(start, t, color=_color_of(prev_reg), alpha=0.15)
            prev_reg = reg; start = t
    if start is not None:
        ax.axvspan(start, day_df.index[-1], color=_color_of(prev_reg), alpha=0.15)
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor="#2ecc71", alpha=0.15, label="GREEN tendencia alcista"),
        Patch(facecolor="#e74c3c", alpha=0.15, label="RED tendencia bajista"),
        Patch(facecolor="#f1c40f", alpha=0.15, label="YELLOW lateral"),
    ]
    ax.legend(handles=legend_patches, loc="upper left")
    ax.set_title(f"{ticker} - regime intraday status (today, smoot) - ADX>={ADX_TREND_MIN}, |slope|>=p{SLOPE_PCTL}, umbral slope: {slope_thr:.2e}", pad=8)
    ax.set_ylabel("Price (USD)")
    ax.grid(True, linestyle="--", alpha=0.25)
    _dates_concise(ax)
    return fig

# =======================
# Cross-asset + Risk-On (figs)
# =======================
def _annotate_last(ax, x, y, label):
    if len(x) == 0: return
    ax.annotate(label, xy=(x[-1], y[-1]), xytext=(5, 0),
                textcoords="offset points", va="center", fontsize=9)

def rolling_beta(sp_ret: pd.Series, asset_ret: pd.Series, win: int) -> pd.Series:
    cov = asset_ret.rolling(win).cov(sp_ret)
    var = sp_ret.rolling(win).var()
    return cov / (var + 1e-12)

def _dominance_color(name: str) -> str:
    if name == "SP": return "#2ecc71"
    if name == "BTC-USD": return "#e74c3c"  # rojo
    if name == "GOLD": return "#d4af37"     # dorado
    if name == "OIL":  return "#7f8c8d"     # gris
    return "#bdc3c7"

def _zscore(series: pd.Series, win: int) -> pd.Series:
    mu = series.rolling(win).mean()
    sd = series.rolling(win).std()
    return (series - mu) / (sd + 1e-12)

def make_cross_asset_figs(sp_bench, assets, period_daily, interval_daily,
                          roll_corr_win, roll_beta_win, dom_ret_win,
                          risk_z_win, risk_ema_span):
    # descarga
    tk_sp, sp = load_any([sp_bench, "SPY"], period_daily, interval_daily)
    if sp.empty:
        return None, None, None, None, None
    loaded = {"SP": (tk_sp, sp[["close"]].rename(columns={"close":"SP"}))}
    for name, tks in assets.items():
        tk, df = load_any(tks, period_daily, interval_daily)
        if not df.empty:
            loaded[name] = (tk, df[["close"]].rename(columns={"close":name}))
    dfs = [v[1] for v in loaded.values()]
    common = dfs[0].join(dfs[1:], how="inner")
    if common.empty:
        return None, None, None, None, None

    # 1) rendimiento acumulado
    norm_pct = 100 * (common / common.iloc[0] - 1.0)
    fig1, ax = plt.subplots(figsize=(12,4))
    for col in norm_pct.columns:
        ax.plot(norm_pct.index, norm_pct[col], label=f"{col}")
        _annotate_last(ax, norm_pct.index, norm_pct[col].values, f"{col} {norm_pct[col].iloc[-1]:+.1f}%")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    loc = mdates.AutoDateLocator(minticks=5, maxticks=8)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax.set_title("Rendimiento acumulado (%) desde inicio de ventana")
    ax.set_ylabel("%"); ax.grid(True, linestyle="--", alpha=0.3); ax.legend(loc="best")
    fig1.tight_layout()

    # 2) corr & 3) beta
    rets = common.pct_change()
    corr_roll, beta_df = pd.DataFrame(index=common.index), pd.DataFrame(index=common.index)
    for col in common.columns:
        if col == "SP": continue
        corr_roll[col] = rets["SP"].rolling(roll_corr_win).corr(rets[col])
        beta_df[col]   = rolling_beta(rets["SP"], rets[col], roll_beta_win)

    fig2 = fig3 = None
    if not corr_roll.empty:
        fig2, ax2 = plt.subplots(figsize=(12,3.8))
        for col in corr_roll.columns:
            ax2.plot(corr_roll.index, corr_roll[col], label=f"{col}")
            last_val = corr_roll[col].dropna().iloc[-1] if corr_roll[col].notna().any() else np.nan
            if not np.isnan(last_val):
                _annotate_last(ax2, corr_roll.index, corr_roll[col].values, f"{col} {last_val:+.2f}")
        for y in [1.0, 0.5, 0.0, -0.5, -1.0]:
            ax2.axhline(y, linestyle="--" if y in (1.0, 0.0, -1.0) else ":", linewidth=1, alpha=0.6 if y in (1.0,0.0,-1.0) else 0.5)
        ax2.set_ylim(-1.05, 1.05)
        loc2 = mdates.AutoDateLocator(minticks=4, maxticks=7)
        ax2.xaxis.set_major_locator(loc2)
        ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc2))
        ax2.set_title(f"Correlation {roll_corr_win}d con SP")
        ax2.set_ylabel("corr"); ax2.grid(True, linestyle="--", alpha=0.3); ax2.legend(loc="best")
        fig2.tight_layout()

    if not beta_df.empty:
        fig3, ax3 = plt.subplots(figsize=(12,3.8))
        for col in beta_df.columns:
            ax3.plot(beta_df.index, beta_df[col], label=f"{col}")
            last_b = beta_df[col].dropna().iloc[-1] if beta_df[col].notna().any() else np.nan
            if not np.isnan(last_b):
                _annotate_last(ax3, beta_df.index, beta_df[col].values, f"{col} {last_b:+.2f}")
        ax3.axhline(0.0, linestyle="--", alpha=0.6)
        ax3.axhline(1.0, linestyle="--", alpha=0.6)
        ymin = min(-0.5, np.nanmin(beta_df.values)-0.2)
        ymax = max(1.5, np.nanmax(beta_df.values)+0.2)
        ax3.set_ylim(ymin, ymax)
        loc3 = mdates.AutoDateLocator(minticks=4, maxticks=7)
        ax3.xaxis.set_major_locator(loc3)
        ax3.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc3))
        ax3.set_title(f"Beta rodante {roll_beta_win}d vs SP")
        ax3.set_ylabel("beta"); ax3.grid(True, linestyle="--", alpha=0.3); ax3.legend(loc="best")
        fig3.tight_layout()

    # 4) Dominancia
    r30 = 100 * common.pct_change(dom_ret_win)
    cols = [c for c in ["SP","BTC-USD","GOLD","OIL"] if c in r30.columns]
    fig4 = None
    if len(cols) >= 2:
        r30_sel = r30[cols].dropna(how="all")
        winner = r30_sel.idxmax(axis=1)
        fig4, ax4 = plt.subplots(figsize=(12, 2.1))
        start = None; prev = None
        for t, w in winner.items():
            if prev is None:
                prev = w; start = t
            elif w != prev:
                ax4.axvspan(start, t, color=_dominance_color(prev), alpha=0.8)
                prev = w; start = t
        if start is not None:
            ax4.axvspan(start, winner.index[-1], color=_dominance_color(prev), alpha=0.8)
        if "SP" in r30_sel.columns:
            ax5 = ax4.twinx()
            ax5.plot(r30_sel.index, r30_sel["SP"], linewidth=1.0, alpha=0.85, color="#2ecc71")
            ax5.set_ylabel("SP 30d ret (%)"); ax5.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax4.set_yticks([])
        loc4 = mdates.AutoDateLocator(minticks=4, maxticks=7)
        ax4.xaxis.set_major_locator(loc4)
        ax4.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc4))
        ax4.set_title("Dominance 30d (color = activo con mayor retorno 30d)")
        from matplotlib.patches import Patch
        legend = [Patch(color=_dominance_color(name), label=name) for name in cols]
        ax4.legend(handles=legend, ncol=len(legend), loc="upper left", framealpha=0.6)
        fig4.tight_layout()

    # 5) Risk-On Score
    z = {}
    for c in ["SP","BTC-USD","GOLD","OIL"]:
        if c in r30.columns:
            z[c] = _zscore(r30[c], risk_z_win)
    terms_pos = [z[k] for k in ["SP","BTC-USD"] if k in z]
    terms_neg = [z[k] for k in ["GOLD","OIL"] if k in z]
    fig5 = None; risk_label = None; risk_val = None
    if len(terms_pos) + len(terms_neg) >= 2:
        num = None
        for s in terms_pos: num = s if num is None else (num + s)
        for s in terms_neg: num = num - s if num is not None else (-s)
        denom = len(terms_pos) + len(terms_neg)
        risk_on = (num / max(denom, 1)).rename("risk_on")
        risk_on_smooth = risk_on.ewm(span=risk_ema_span, adjust=False).mean()
        fig5, ax6 = plt.subplots(figsize=(12,3.8))
        ax6.plot(risk_on.index, risk_on, linewidth=1.0, label="risk_on (z-avg)")
        ax6.plot(risk_on_smooth.index, risk_on_smooth, linewidth=1.8, label=f"EMA{risk_ema_span}")
        ax6.axhline(0.0, linestyle="--", alpha=0.6)
        ax6.axhline(0.5, linestyle=":", alpha=0.8)
        ax6.axhline(-0.5, linestyle=":", alpha=0.8)
        last_val = risk_on.dropna().iloc[-1]
        risk_val = float(last_val)
        risk_label = "RISK-ON" if last_val > 0.5 else ("RISK-OFF" if last_val < -0.5 else "NEUTRAL")
        _annotate_last(ax6, risk_on.index, risk_on.values, f"{risk_label} {last_val:+.2f}")
        loc5 = mdates.AutoDateLocator(minticks=4, maxticks=7)
        ax6.xaxis.set_major_locator(loc5)
        ax6.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc5))
        ax6.set_title("Risk-On Score (z(SP30d)+z(BTC30d)-z(GOLD30d)-z(OIL30d))")
        ax6.set_ylabel("z-score medio")
        ax6.grid(True, linestyle="--", alpha=0.3)
        ax6.legend()
        fig5.tight_layout()

    return fig1, fig2, fig3, fig4, fig5, risk_label, risk_val, r30

# =======================
# App logic
# =======================
# 1) intradia + semanal
ticker, df = load_any([TICKER_PRI, "^GSPC"], PERIOD_INTRADAY, INTERVAL_INTRADAY)
ticker_w, df_w = load_any([ticker] if ticker else [TICKER_PRI, "^GSPC"], PERIOD_WEEKLY_STRIP, INTERVAL_WEEKLY_STRIP)

if df.empty:
    st.error("Data not available. Revise conextion /market/interval.")
    st.stop()

df = compute_regime(df)
summary, day_df = summarize_today_with_combined_rule(df)

weekly_df = compute_regime(df_w) if df_w is not None and not df_w.empty else None
slope_thr = df.attrs.get("slope_thr", 0.0)

with st.container():
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("regime intraday + week")
        fig = make_fig_semaforo(day_df, weekly_df, ticker, slope_thr, WEEKLY_WINDOW_DAYS)
        st.pyplot(fig, clear_figure=True)
    with col2:
        st.markdown("### Resume intraday")
        st.write(f"**Ticker:** {ticker}")
        st.write(f"**Sesion:** {summary['date']}")
        st.write(f"**Regime actual (ultima):** {summary['last_regime']} (run={summary['last_run']})")
        st.write(f"**Regime dominant today:** {summary['dominant_regime']}")
        st.write(f"**Regime FINAL:** {summary['final_regime']}  \n↳ {summary['decision_reason']}")
        st.write(f"**Return today:** {summary['ret_today_pct']:.2f}%")
        if not math.isnan(summary['vol_today_pct']):
            st.write(f"**Vol. realizada:** {summary['vol_today_pct']:.2f}%")
        st.write(f"**Status regime :** GREEN {summary['pct_green']:.1f}% | YELLOW {summary['pct_yellow']:.1f}% | RED {summary['pct_red']:.1f}%")
        st.caption(f"Parameters: ADX_WINDOW={ADX_WINDOW}, SLOPE_WINDOW={SLOPE_WINDOW}, ADX_TREND_MIN={ADX_TREND_MIN}, slope pctl={SLOPE_PCTL}")

# 2) cross-asset + risk-on
if RUN_CROSS_ASSET:
    ASSETS = {
        "BTC-USD": ["BTC-USD"],
        "GOLD":    ["GLD", "GC=F"],
        "OIL":     ["CL=F", "USO"],
    }
    figs = make_cross_asset_figs(SP_BENCH, ASSETS, PERIOD_DAILY, INTERVAL_DAILY,
                                 ROLL_CORR_WIN, ROLL_BETA_WIN, DOM_RET_WIN,
                                 RISK_Z_WIN, RISK_EMA_SPAN)
    fig1, fig2, fig3, fig4, fig5, risk_label, risk_val, r30 = figs
    st.subheader("Comparison cross-asset (daily)")
    if fig1: st.pyplot(fig1, clear_figure=True)
    if fig2: st.pyplot(fig2, clear_figure=True)
    if fig3: st.pyplot(fig3, clear_figure=True)
    if fig4: st.pyplot(fig4, clear_figure=True)
    if fig5:
        st.pyplot(fig5, clear_figure=True)
        st.info(f"**Risk-On Score:** {risk_label} ({risk_val:+.2f})  \nDef: z(SP30d)+z(BTC30d)-z(GOLD30d)-z(OIL30d) | z-win={RISK_Z_WIN}, EMA={RISK_EMA_SPAN}")

    # Resumen numerico (última observacion)
    if r30 is not None:
        st.markdown("### Resumen numerico (ultimos valores)")
        cols = [c for c in ["SP","BTC-USD","GOLD","OIL"] if c in r30.columns]
        if cols:
            last_row = r30[cols].iloc[-1].round(2)
            for k, v in last_row.items():
                st.write(f"- **{k:8s}** ret30d: {v: .2f}%")
