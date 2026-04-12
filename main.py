import math
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go


# BUS6440 dashboard main file
# This version uses live DAILY data from Yahoo Finance.
# The project started from spreadsheet-style analysis, then moved into Python/Dash
# once rolling metrics, event windows, and forecasting became easier to manage in code.

# I kept the start date wide enough to capture multiple market regimes.
# That helps the dashboard compare calm periods, crisis windows, and recoveries.
DATA_START = "2006-01-01"
APP_TITLE = "BUS6440 | Gold & Silver Safety Dashboard"
TRADING_DAYS = 252

TICKERS = {
    "gold": "GC=F",
    "silver": "SI=F",
    "sp500": "^GSPC",
    "usdcad": "USDCAD=X",
}

CRISIS_EVENTS = [
    {"label": "GFC 2008", "start": "2008-09", "end": "2008-12"},
    {"label": "COVID 2020", "start": "2020-02", "end": "2020-04"},
    {"label": "Rate Shock 2022", "start": "2022-01", "end": "2022-10"},
    {"label": "2025 Bull Run", "start": "2025-01", "end": "2025-06"},
]

# Annual CPI values used only for rough inflation-adjusted visual comparison.
# For a more formal paper, these should be replaced with a sourced CPI series file.
CPI = {
    2006: 201.6, 2007: 207.3, 2008: 215.3, 2009: 214.5, 2010: 218.1,
    2011: 224.9, 2012: 229.6, 2013: 233.0, 2014: 236.7, 2015: 237.0,
    2016: 240.0, 2017: 245.1, 2018: 251.1, 2019: 255.7, 2020: 258.8,
    2021: 271.0, 2022: 296.8, 2023: 304.7, 2024: 313.5, 2025: 320.0,
    2026: 326.0,
}

TBILL_ANN = 0.0473
TBILL_DAY = (1 + TBILL_ANN) ** (1 / TRADING_DAYS) - 1

BG = "#0F1923"
CARD = "#182330"
GRID = "#2A3C52"
WHITE = "#F1F5F9"
MUTED = "#94A3B8"
GOLD_C = "#D4AF37"
SILVER_C = "#A0AEC0"
EQUITY_C = "#22C55E"
AMBER = "#F59E0B"
RED = "#EF4444"


# Each series is downloaded live so the dashboard feels current at launch.
# I used adjusted close because it is the cleanest single price field to work with.
def download_daily_series(ticker: str) -> pd.Series:
    df = yf.download(
        ticker,
        start=DATA_START,
        interval="1d",
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")

    series = df["Close"].squeeze().dropna()
    series.index = pd.to_datetime(series.index)
    series.index = series.index.strftime("%Y-%m-%d")
    return series


# This is the main data loading step.
# Gold, silver, equities, and FX are aligned together so the later comparisons are apples-to-apples.
def load_market_data() -> pd.DataFrame:
    print("Fetching live daily data from Yahoo Finance...")

    gold = download_daily_series(TICKERS["gold"])
    silver = download_daily_series(TICKERS["silver"])
    sp500 = download_daily_series(TICKERS["sp500"])
    usdcad = download_daily_series(TICKERS["usdcad"])

    df = pd.DataFrame(
        {
            "gold_usd": gold,
            "silver_usd": silver,
            "sp500_usd": sp500,
            "usdcad": usdcad,
        }
    ).ffill().dropna()

    df["gold_cad"] = df["gold_usd"] * df["usdcad"]
    df["silver_cad"] = df["silver_usd"] * df["usdcad"]
    df["sp500_cad"] = df["sp500_usd"] * df["usdcad"]

    print(f"Loaded {len(df):,} daily observations from {df.index[0]} to {df.index[-1]}")
    return df.round(4)


# Daily returns are the base unit for most of the analytics below.
# Once returns are stable, annualised metrics and rolling comparisons become much easier to explain.
def daily_returns(prices) -> np.ndarray:
    p = np.array(prices, dtype=float)
    return np.diff(p) / p[:-1]


# This function is the dashboard's risk summary engine.
# I grouped the core measures together because these are the ones people usually ask about first:
# volatility, return quality, downside pain, and worst historical loss.
def risk_metrics(prices, rf_daily=TBILL_DAY) -> dict:
    ret = daily_returns(prices)
    if len(ret) == 0:
        return {}

    vol = float(np.std(ret, ddof=1) * math.sqrt(TRADING_DAYS))
    mean_ret = float(np.mean(ret))
    ann_ret = float((1 + mean_ret) ** TRADING_DAYS - 1)
    excess = ret - rf_daily

    sharpe = 0.0
    if np.std(excess, ddof=1) > 0:
        sharpe = float(np.mean(excess) / np.std(excess, ddof=1) * math.sqrt(TRADING_DAYS))

    downside = excess[excess < 0]
    sortino = 0.0
    if len(downside) > 1 and np.std(downside, ddof=1) > 0:
        sortino = float(np.mean(excess) / (np.std(downside, ddof=1) * math.sqrt(TRADING_DAYS)))

    cum = np.cumprod(1 + ret)
    peak = np.maximum.accumulate(cum)
    dd_series = (cum - peak) / peak
    max_dd = float(np.min(dd_series))
    calmar = float(ann_ret / abs(max_dd)) if max_dd != 0 else 0.0

    n = len(prices)
    cagr = float((prices[-1] / prices[0]) ** (TRADING_DAYS / n) - 1) if n > 1 else 0.0

    return {
        "vol": vol,
        "ann_ret": ann_ret,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "calmar": calmar,
        "cagr": cagr,
    }


# I used a 90-day window here because very short windows became too jumpy,
# while very long windows hid meaningful regime shifts.
def rolling_correlation(r1, r2, window=90):
    n = min(len(r1), len(r2))
    a = np.array(r1[:n])
    b = np.array(r2[:n])
    out = [None] * (window - 1)
    for i in range(window - 1, n):
        out.append(float(np.corrcoef(a[i - window + 1:i + 1], b[i - window + 1:i + 1])[0, 1]))
    return out


# CAPM is included here as a simple way to show market sensitivity.
# It is not the full story of safety, but it helps explain how tightly each asset moves with equities.
def capm_regression(asset_returns, market_returns):
    n = min(len(asset_returns), len(market_returns))
    x = np.array(market_returns[:n])
    y = np.array(asset_returns[:n])
    slope, intercept, r, _, _ = stats.linregress(x, y)
    return {"beta": float(slope), "alpha": float(intercept), "r2": float(r ** 2)}


def log_linear_trend(prices):
    p = np.array(prices, dtype=float)
    x = np.arange(len(p))
    slope, intercept, r, _, _ = stats.linregress(x, np.log(p))
    return intercept, slope, float(r ** 2)


# I kept the forecast intentionally simple with AR(3).
# The goal here is not to pretend we can predict markets perfectly,
# but to show short-memory behaviour in returns in a transparent way.
def ar_forecast(returns, p=3, steps=30):
    n = len(returns)
    if n <= p:
        return [], [0.0] * steps

    X = np.array([returns[i - p:i][::-1] for i in range(p, n)])
    y = np.array([returns[i] for i in range(p, n)])
    coeffs, _, _, _ = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)

    intercept = coeffs[0]
    phi = coeffs[1:]

    fitted = []
    for i in range(p, n):
        pred = intercept + sum(phi[j] * returns[i - 1 - j] for j in range(p))
        fitted.append(pred)

    last_window = list(returns[-p:][::-1])
    forecast = []
    for _ in range(steps):
        pred = intercept + sum(phi[j] * last_window[j] for j in range(p))
        forecast.append(pred)
        last_window = [pred] + last_window[:-1]

    return fitted, forecast


def inflation_adjust(prices, dates):
    base_cpi = CPI[2006]
    adjusted = []
    for price, date_str in zip(prices, dates):
        year = int(str(date_str)[:4])
        cpi = CPI.get(year, list(CPI.values())[-1])
        adjusted.append(price * base_cpi / cpi)
    return adjusted


def next_business_day(date_str: str, offset: int = 1) -> str:
    ts = pd.Timestamp(date_str)
    return (ts + pd.offsets.BDay(offset)).strftime("%Y-%m-%d")


def nearest_date_idx(dates, date_str: str) -> int:
    if date_str in dates:
        return dates.index(date_str)

    if len(date_str) == 7:
        target = pd.Timestamp(f"{date_str}-01")
    else:
        target = pd.Timestamp(date_str)

    parsed = [pd.Timestamp(x) for x in dates]
    return min(range(len(parsed)), key=lambda i: abs((parsed[i] - target).days))


def base_layout(title: str):
    return dict(
        title=title,
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(color=WHITE, family="Arial, sans-serif"),
        margin=dict(l=50, r=20, t=60, b=50),
        hovermode="x unified",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor=GRID, linecolor=GRID),
        yaxis=dict(gridcolor=GRID, linecolor=GRID),
    )


def get_price_columns(currency: str):
    suffix = "usd" if currency == "USD" else "cad"
    return f"gold_{suffix}", f"silver_{suffix}", f"sp500_{suffix}"


# OVERVIEW TAB
# This is the first impression chart.
# I normalised everything to 100 so users can compare direction and growth without getting distracted by different price scales.
def build_overview(df: pd.DataFrame, currency: str):
    g_col, s_col, e_col = get_price_columns(currency)
    dates = df.index.to_list()

    g = df[g_col].to_numpy()
    s = df[s_col].to_numpy()
    e = df[e_col].to_numpy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=g / g[0] * 100, name="Gold", line=dict(color=GOLD_C, width=2.5)))
    fig.add_trace(go.Scatter(x=dates, y=s / s[0] * 100, name="Silver", line=dict(color=SILVER_C, width=2.5)))
    fig.add_trace(go.Scatter(x=dates, y=e / e[0] * 100, name="S&P 500", line=dict(color=EQUITY_C, width=2.5)))
    fig.update_layout(**base_layout(f"Normalised Performance (Base = 100) | {currency}"))
    return fig


# OVERVIEW TAB - REAL GROWTH SECTION
# This section answers a more practical question: what happened after inflation pressure is considered?
# It helps move the discussion from headline returns to purchasing-power style thinking.
def build_real_growth(df: pd.DataFrame, currency: str):
    g_col, s_col, e_col = get_price_columns(currency)
    dates = df.index.to_list()

    g = np.array(inflation_adjust(df[g_col].to_list(), dates))
    s = np.array(inflation_adjust(df[s_col].to_list(), dates))
    e = np.array(inflation_adjust(df[e_col].to_list(), dates))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=g / g[0] * 100, name="Gold (real)", line=dict(color=GOLD_C, width=2.5)))
    fig.add_trace(go.Scatter(x=dates, y=s / s[0] * 100, name="Silver (real)", line=dict(color=SILVER_C, width=2.5)))
    fig.add_trace(go.Scatter(x=dates, y=e / e[0] * 100, name="S&P 500 (real)", line=dict(color=EQUITY_C, width=2.5)))
    fig.update_layout(**base_layout(f"Inflation-Adjusted Growth | {currency}"))
    return fig


# RISK TAB - SUMMARY SECTION
# I used Sharpe first because it is familiar and easy to explain in a presentation.
# It gives a quick read on how much return quality each asset delivered per unit of volatility.
def build_risk_bars(df: pd.DataFrame, currency: str):
    g_col, s_col, e_col = get_price_columns(currency)

    g_metrics = risk_metrics(df[g_col].to_list())
    s_metrics = risk_metrics(df[s_col].to_list())
    e_metrics = risk_metrics(df[e_col].to_list())

    labels = ["Gold", "Silver", "S&P 500"]
    values = [g_metrics["sharpe"], s_metrics["sharpe"], e_metrics["sharpe"]]
    colors = [GOLD_C, SILVER_C, EQUITY_C]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"{v:.2f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(**base_layout(f"Sharpe Ratio Comparison | {currency}"))
    return fig


# RISK TAB - DRAWDOWN SECTION
# This is one of the most important safety views in the whole dashboard.
# A defensive asset should not only grow over time, it should also lose less when markets become stressed.
def build_drawdown(df: pd.DataFrame, currency: str):
    g_col, s_col, e_col = get_price_columns(currency)
    dates = df.index.to_list()[1:]

    def dd_series(prices):
        ret = daily_returns(prices)
        cum = np.cumprod(1 + ret)
        peak = np.maximum.accumulate(cum)
        return (cum - peak) / peak * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=dd_series(df[g_col].to_list()), name="Gold", fill="tozeroy", line=dict(color=GOLD_C, width=1.5)))
    fig.add_trace(go.Scatter(x=dates, y=dd_series(df[s_col].to_list()), name="Silver", fill="tozeroy", line=dict(color=SILVER_C, width=1.5)))
    fig.add_trace(go.Scatter(x=dates, y=dd_series(df[e_col].to_list()), name="S&P 500", fill="tozeroy", line=dict(color=EQUITY_C, width=1.5)))
    fig.update_layout(**base_layout(f"Drawdown from Peak (%) | {currency}"))
    return fig


# CORRELATION TAB - ROLLING SECTION
# I wanted this tab to show that relationships are not static.
# Assets can look like diversifiers in one period and move more like risk assets in another.
def build_rolling_corr(df: pd.DataFrame):
    g_ret = daily_returns(df["gold_usd"].to_list())
    s_ret = daily_returns(df["silver_usd"].to_list())
    e_ret = daily_returns(df["sp500_usd"].to_list())
    dates = df.index.to_list()[1:]

    g_sp = rolling_correlation(g_ret, e_ret, window=90)
    s_sp = rolling_correlation(s_ret, e_ret, window=90)

    fig = go.Figure()
    fig.add_hline(y=0, line_color=MUTED, line_dash="dot")
    fig.add_trace(go.Scatter(x=dates, y=g_sp, name="Gold vs S&P 500", line=dict(color=GOLD_C, width=2)))
    fig.add_trace(go.Scatter(x=dates, y=s_sp, name="Silver vs S&P 500", line=dict(color=SILVER_C, width=2)))
    fig.update_layout(**base_layout("90-Day Rolling Correlation"))
    fig.update_yaxes(range=[-1, 1])
    return fig


# CORRELATION TAB - CAPM SECTION
# This chart gives a more formal market-link view beside the rolling correlation chart.
# Beta helps explain whether gold or silver behaves more defensively relative to the equity market.
def build_capm_chart(df: pd.DataFrame):
    g_ret = daily_returns(df["gold_usd"].to_list())
    s_ret = daily_returns(df["silver_usd"].to_list())
    e_ret = daily_returns(df["sp500_usd"].to_list())

    g_capm = capm_regression(g_ret, e_ret)
    s_capm = capm_regression(s_ret, e_ret)
    x_line = np.linspace(np.min(e_ret), np.max(e_ret), 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=e_ret, y=g_ret, mode="markers", name="Gold", marker=dict(color=GOLD_C, size=6, opacity=0.5)))
    fig.add_trace(go.Scatter(x=x_line, y=g_capm["alpha"] + g_capm["beta"] * x_line, name=f"Gold fit (β={g_capm['beta']:.2f})", line=dict(color=GOLD_C, width=2)))
    fig.add_trace(go.Scatter(x=e_ret, y=s_ret, mode="markers", name="Silver", marker=dict(color=SILVER_C, size=6, opacity=0.5)))
    fig.add_trace(go.Scatter(x=x_line, y=s_capm["alpha"] + s_capm["beta"] * x_line, name=f"Silver fit (β={s_capm['beta']:.2f})", line=dict(color=SILVER_C, width=2)))
    fig.update_layout(**base_layout("CAPM Regression vs S&P 500"))
    fig.update_xaxes(title="S&P 500 Daily Return")
    fig.update_yaxes(title="Asset Daily Return")
    return fig


# EVENTS TAB
# I added event windows because averages alone can hide what really matters in crisis periods.
# This section lets the user test the safety story exactly when protection was supposed to matter most.
def build_event_bar(df: pd.DataFrame, event_label: str):
    dates = df.index.to_list()
    event = next(x for x in CRISIS_EVENTS if x["label"] == event_label)
    start_idx = nearest_date_idx(dates, event["start"])
    end_idx = nearest_date_idx(dates, event["end"])

    def total_return(series):
        prices = series.to_list()
        return (prices[end_idx] / prices[start_idx] - 1) * 100

    vals = [
        total_return(df["gold_usd"]),
        total_return(df["silver_usd"]),
        total_return(df["sp500_usd"]),
    ]

    colors = [
        GOLD_C if vals[0] >= 0 else RED,
        SILVER_C if vals[1] >= 0 else RED,
        EQUITY_C if vals[2] >= 0 else RED,
    ]

    fig = go.Figure(go.Bar(
        x=["Gold", "Silver", "S&P 500"],
        y=vals,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in vals],
        textposition="outside",
    ))
    fig.update_layout(**base_layout(f"Total Return During {event_label}"))
    fig.update_yaxes(title="Return (%)")
    return fig


# FORECAST TAB
# This tab is more exploratory than definitive.
# I included both a fitted trend and a simple AR forecast so users can compare longer drift versus short-run momentum or mean reversion.
def build_forecast_chart(df: pd.DataFrame, asset: str, horizon: int = 30):
    column = "gold_usd" if asset == "Gold" else "silver_usd"
    color = GOLD_C if asset == "Gold" else SILVER_C
    prices = df[column].to_list()
    dates = df.index.to_list()

    returns = daily_returns(prices)
    fitted_ret, forecast_ret = ar_forecast(returns, p=3, steps=horizon)

    fitted_prices = [prices[3]]
    for r in fitted_ret:
        fitted_prices.append(fitted_prices[-1] * (1 + r))

    forecast_prices = [prices[-1]]
    for r in forecast_ret:
        forecast_prices.append(forecast_prices[-1] * (1 + r))

    fitted_dates = dates[3:3 + len(fitted_prices)]
    future_dates = [dates[-1]] + [next_business_day(dates[-1], i) for i in range(1, horizon + 1)]

    a, b, r2 = log_linear_trend(prices)
    trend = [math.exp(a + b * i) for i in range(len(prices))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=prices, name=f"{asset} actual", line=dict(color=color, width=2.5)))
    fig.add_trace(go.Scatter(x=dates, y=trend, name=f"Linear trend (R²={r2:.2f})", line=dict(color=MUTED, width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=fitted_dates, y=fitted_prices, name="AR(3) fitted", line=dict(color=AMBER, width=1.5, dash="dash")))
    fig.add_trace(go.Scatter(x=future_dates, y=forecast_prices, name=f"AR(3) forecast ({horizon} days)", line=dict(color=RED, width=2)))
    fig.update_layout(**base_layout(f"{asset} Forecast"))
    return fig


# PORTFOLIO TAB
# This section is where the analysis becomes more decision-oriented.
# Instead of judging each asset alone, it shows how mixes of gold, silver, and equities may behave together.
def build_monte_carlo(df: pd.DataFrame, gold_weight: float, silver_weight: float, equity_weight: float, steps: int = 60, paths: int = 300):
    total = gold_weight + silver_weight + equity_weight
    wg = gold_weight / total
    ws = silver_weight / total
    we = equity_weight / total

    g_ret = daily_returns(df["gold_usd"].to_list())
    s_ret = daily_returns(df["silver_usd"].to_list())
    e_ret = daily_returns(df["sp500_usd"].to_list())
    n = min(len(g_ret), len(s_ret), len(e_ret))

    port_ret = wg * np.array(g_ret[:n]) + ws * np.array(s_ret[:n]) + we * np.array(e_ret[:n])
    mu = float(np.mean(port_ret))
    sig = float(np.std(port_ret, ddof=1))

    start_value = 100.0
    future_dates = [df.index[-1]] + [next_business_day(df.index[-1], i) for i in range(1, steps + 1)]

    np.random.seed(42)
    simulated = []
    for _ in range(paths):
        path = [start_value]
        for _ in range(steps):
            path.append(path[-1] * (1 + np.random.normal(mu, sig)))
        simulated.append(path)

    arr = np.array(simulated)
    p10 = np.percentile(arr, 10, axis=0)
    p50 = np.percentile(arr, 50, axis=0)
    p90 = np.percentile(arr, 90, axis=0)

    fig = go.Figure()
    for path in simulated[:60]:
        fig.add_trace(go.Scatter(x=future_dates, y=path, mode="lines", line=dict(color="rgba(212,175,55,0.05)", width=1), showlegend=False))

    fig.add_trace(go.Scatter(
        x=future_dates + future_dates[::-1],
        y=list(p10) + list(p90[::-1]),
        fill="toself",
        fillcolor="rgba(245,158,11,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="P10-P90",
    ))
    fig.add_trace(go.Scatter(x=future_dates, y=p50, name="Median", line=dict(color=GOLD_C, width=2.5)))
    fig.add_trace(go.Scatter(x=future_dates, y=p10, name="P10", line=dict(color=RED, width=1.5, dash="dash")))
    fig.add_trace(go.Scatter(x=future_dates, y=p90, name="P90", line=dict(color=EQUITY_C, width=1.5, dash="dash")))
    fig.update_layout(**base_layout("Portfolio Monte Carlo Simulation"))
    return fig


def stat_cards(df: pd.DataFrame):
    g = risk_metrics(df["gold_usd"].to_list())
    s = risk_metrics(df["silver_usd"].to_list())
    e = risk_metrics(df["sp500_usd"].to_list())

    cards = [
        ("Gold CAGR", f"{g['cagr'] * 100:.1f}%"),
        ("Silver CAGR", f"{s['cagr'] * 100:.1f}%"),
        ("S&P CAGR", f"{e['cagr'] * 100:.1f}%"),
        ("Data Range", f"{df.index[0]} to {df.index[-1]}"),
    ]

    return html.Div(
        [
            html.Div(
                [
                    html.Div(title, style={"fontSize": "12px", "color": MUTED}),
                    html.Div(value, style={"fontSize": "20px", "fontWeight": "700", "marginTop": "6px"}),
                ],
                style={
                    "backgroundColor": CARD,
                    "padding": "14px",
                    "borderRadius": "12px",
                    "border": f"1px solid {GRID}",
                    "flex": "1",
                    "minWidth": "180px",
                },
            )
            for title, value in cards
        ],
        style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "18px"},
    )


DF = load_market_data()
FETCHED_AT = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

app = Dash(__name__)
server = app.server
app.title = APP_TITLE

# App layout is kept simple on purpose.
# I wanted the controls at the top so users can switch currency, event, and asset without hunting through the screen.
app.layout = html.Div(
    style={
        "backgroundColor": BG,
        "minHeight": "100vh",
        "padding": "24px",
        "fontFamily": "Arial, sans-serif",
        "color": WHITE,
    },
    children=[
        html.H1("Gold & Silver Safety Dashboard", style={"marginBottom": "4px"}),
        html.Div(
            f"BUS6440 project | Live DAILY data from Yahoo Finance | Fetched {FETCHED_AT}",
            style={"color": MUTED, "marginBottom": "20px"},
        ),
        stat_cards(DF),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Currency", style={"display": "block", "marginBottom": "6px"}),
                        dcc.Dropdown(["USD", "CAD"], "USD", id="currency", clearable=False),
                    ],
                    style={"flex": "1", "minWidth": "180px"},
                ),
                html.Div(
                    [
                        html.Label("Crisis Event", style={"display": "block", "marginBottom": "6px"}),
                        dcc.Dropdown([x["label"] for x in CRISIS_EVENTS], CRISIS_EVENTS[0]["label"], id="event", clearable=False),
                    ],
                    style={"flex": "1", "minWidth": "220px"},
                ),
                html.Div(
                    [
                        html.Label("Forecast Asset", style={"display": "block", "marginBottom": "6px"}),
                        dcc.Dropdown(["Gold", "Silver"], "Gold", id="forecast_asset", clearable=False),
                    ],
                    style={"flex": "1", "minWidth": "180px"},
                ),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "20px"},
        ),
        dcc.Tabs(
            id="tabs",
            value="overview",
            children=[
                dcc.Tab(label="Overview", value="overview", style={"backgroundColor": CARD, "color": WHITE}, selected_style={"backgroundColor": GRID, "color": WHITE}),
                dcc.Tab(label="Risk", value="risk", style={"backgroundColor": CARD, "color": WHITE}, selected_style={"backgroundColor": GRID, "color": WHITE}),
                dcc.Tab(label="Correlation", value="correlation", style={"backgroundColor": CARD, "color": WHITE}, selected_style={"backgroundColor": GRID, "color": WHITE}),
                dcc.Tab(label="Events", value="events", style={"backgroundColor": CARD, "color": WHITE}, selected_style={"backgroundColor": GRID, "color": WHITE}),
                dcc.Tab(label="Forecast", value="forecast", style={"backgroundColor": CARD, "color": WHITE}, selected_style={"backgroundColor": GRID, "color": WHITE}),
                dcc.Tab(label="Portfolio", value="portfolio", style={"backgroundColor": CARD, "color": WHITE}, selected_style={"backgroundColor": GRID, "color": WHITE}),
            ],
        ),
        html.Div(id="tab-content", style={"marginTop": "18px"}),
    ],
)


@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("currency", "value"),
    Input("event", "value"),
    Input("forecast_asset", "value"),
)
def render_tab(tab, currency, event_label, forecast_asset):
        # Overview tab sets the story foundation before moving into deeper analytics.
    if tab == "overview":
        return html.Div([
            dcc.Graph(figure=build_overview(DF, currency)),
            dcc.Graph(figure=build_real_growth(DF, currency)),
        ])

        # Risk tab focuses on whether the asset was actually defensive, not just profitable.
    if tab == "risk":
        return html.Div([
            dcc.Graph(figure=build_risk_bars(DF, currency)),
            dcc.Graph(figure=build_drawdown(DF, currency)),
        ])

        # Correlation tab checks whether diversification held consistently or only in selected periods.
    if tab == "correlation":
        return html.Div([
            dcc.Graph(figure=build_rolling_corr(DF)),
            dcc.Graph(figure=build_capm_chart(DF)),
        ])

        # Events tab zooms into stress windows because safety matters most when markets break.
    if tab == "events":
        return html.Div([
            dcc.Graph(figure=build_event_bar(DF, event_label)),
            html.Div(
                f"Selected event window: {event_label}. Month-based event labels are mapped to the closest available trading days in the live daily series.",
                style={"color": MUTED, "padding": "8px 4px 0 4px"},
            ),
        ])

        # Forecast tab is included as a learning and exploration layer rather than a promise of prediction.
    if tab == "forecast":
        return html.Div([
            dcc.Graph(figure=build_forecast_chart(DF, forecast_asset, horizon=30)),
            html.Div(
                "Forecast tab uses an AR(3) model on daily returns plus a simple log-linear trend for comparison.",
                style={"color": MUTED, "padding": "8px 4px 0 4px"},
            ),
        ])

        # Portfolio tab turns the analysis into a practical allocation conversation.
    return html.Div([
        dcc.Graph(figure=build_monte_carlo(DF, gold_weight=0.15, silver_weight=0.10, equity_weight=0.75, steps=60, paths=300)),
        html.Div(
            "Portfolio simulation shown here is a simple parametric Monte Carlo using historical daily mean and volatility.",
            style={"color": MUTED, "padding": "8px 4px 0 4px"},
        ),
    ])


if __name__ == "__main__":
    app.run(debug=True)
