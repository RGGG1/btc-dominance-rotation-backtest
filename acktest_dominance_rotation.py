"""
BTC Dominance Rotation Backtest
-------------------------------

Strategy:
- Universe: BTC + top N non-stable altcoins by market cap (today's top, used backward).
- Dominance signal (daily):
    dom(t) = BTC_mcap(t) / (BTC_mcap(t) + SUM alt_mcap_i(t))

- Bands (you can tune):
    lower_band = 0.60
    upper_band = 0.75

- BTC weight:
    if dom <= lower_band: wBTC = 0 (all alts)
    if dom >= upper_band: wBTC = 1 (all BTC)
    else: linear between

- Alts bucket: market-cap–weighted basket of the top N alts.
- Start: DESIRED_START_DATE (but actual start may be clamped by API limits)
    - Split according to wBTC(dom(start)).
- Then each day:
    - Add DAILY_DCA dollars (e.g. $10).
    - Allocate DCA according to today's wBTC.
    - Rebalance to target weights (wBTC vs 1-wBTC).
- Benchmark: same cash-flow, but 100% BTC-only DCA.
- Output: annual summary and full-period results.

NOTE:
- This uses CoinGecko API.
- Set your CoinGecko API key via environment variable COINGECKO_API_KEY.
- Free/public plans only allow historical data for roughly the last 365 days,
  so we clamp the start date accordingly.
"""

import os
import time
from datetime import datetime, timezone, timedelta

import requests
import pandas as pd
import numpy as np

# ---------------- CONFIG ----------------

COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")

if not COINGECKO_API_KEY:
    raise RuntimeError(
        "COINGECKO_API_KEY environment variable is not set. "
        "Set it in your environment or as a GitHub Secret."
    )

BASE_URL = "https://api.coingecko.com/api/v3"

# What you conceptually want as the start
DESIRED_START_DATE = "2023-01-01"
# End date for the analysis
END_DATE           = "2025-11-01"

TOP_N_ALTS = 20  # number of non-stable alts in the alt basket

# Dominance bands (we will likely widen/recalibrate these later)
LOWER_BAND = 0.60
UPPER_BAND = 0.75

INITIAL_CAPITAL = 100.0
DAILY_DCA = 10.0

OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

# Common stablecoin IDs on CoinGecko to exclude from alts:
STABLE_IDS = {
    "tether",          # USDT
    "usd-coin",        # USDC
    "binance-usd",     # BUSD
    "dai",             # DAI
    "true-usd",        # TUSD
    "frax",            # FRAX
    "paxos-standard",  # USDP / PAX
    "usdd",            # USDD
    "first-digital-usd",
    "lusd",            # Liquity USD
    "paxos-usd",
    # Extra ones we saw popping up:
    "usds",
    "binance-bridged-usdt-bnb-smart-chain",
}


# -------------- API HELPERS --------------


def cg_get(path, params=None, sleep_sec=1.2):
    """
    Simple CoinGecko GET wrapper with API key + crude rate limiting.
    Uses the demo/pro key header parameter 'x-cg-demo-api-key'.
    Adjust if your plan uses a different mechanism.
    """
    if params is None:
        params = {}

    headers = {
        "Accept": "application/json",
        "x-cg-demo-api-key": COINGECKO_API_KEY,
    }

    url = BASE_URL + path
    resp = requests.get(url, params=params, headers=headers)
    if resp.status_code != 200:
        raise RuntimeError(f"CoinGecko API error {resp.status_code}: {resp.text[:300]}")

    # crude rate limit spacing
    time.sleep(sleep_sec)
    return resp.json()


def get_top_alts_ex_stables(top_n=20, vs_currency="usd"):
    """
    Get current top non-stable coins (excluding BTC) by market cap from CoinGecko.
    We use today's top list as a fixed universe for the backtest.
    """
    data = cg_get(
        "/coins/markets",
        params={
            "vs_currency": vs_currency,
            "order": "market_cap_desc",
            "per_page": 100,
            "page": 1,
            "sparkline": "false",
        },
    )

    alts = []
    for coin in data:
        cid = coin["id"]
        if cid == "bitcoin":
            continue
        if cid in STABLE_IDS:
            continue
        alts.append(cid)

    return alts[:top_n]


def get_market_chart_range(coin_id, vs_currency, from_ts, to_ts):
    """
    Fetch historical market chart (prices + market caps) for a coin between two
    UNIX timestamps. Returns a DataFrame with daily data.
    """
    js = cg_get(
        f"/coins/{coin_id}/market_chart/range",
        params={
            "vs_currency": vs_currency,
            "from": int(from_ts),
            "to": int(to_ts),
        },
    )

    # js: { prices: [[ts, price], ...], market_caps: [[ts, mcap], ...], ... }
    prices = pd.DataFrame(js["prices"], columns=["ts", "price"])
    mcaps = pd.DataFrame(js["market_caps"], columns=["ts", "mcap"])

    # convert ms -> date
    prices["date"] = pd.to_datetime(prices["ts"], unit="ms").dt.date
    mcaps["date"] = pd.to_datetime(mcaps["ts"], unit="ms").dt.date

    # daily: take last value of the day
    prices = prices.groupby("date")["price"].last().reset_index()
    mcaps = mcaps.groupby("date")["mcap"].last().reset_index()

    df = pd.merge(prices, mcaps, on="date", how="outer").sort_values("date")
    df["coin_id"] = coin_id
    return df


def date_to_unix(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return dt.timestamp()


def get_effective_dates():
    """
    CoinGecko free/public plans typically limit historical range (e.g. last 365 days).
    We clamp the start date so the range does not exceed 365 days.
    """
    desired_start = datetime.strptime(DESIRED_START_DATE, "%Y-%m-%d").date()
    end_date = datetime.strptime(END_DATE, "%Y-%m-%d").date()

    max_span_start = end_date - timedelta(days=365)
    effective_start = max(desired_start, max_span_start)

    return effective_start, end_date


# -------------- STRATEGY LOGIC --------------


def w_btc_from_dom(d):
    """
    BTC weight as function of dominance d:
    - d <= LOWER_BAND: 0 (all alts)
    - d >= UPPER_BAND: 1 (all BTC)
    - linear in between
    """
    if d <= LOWER_BAND:
        return 0.0
    if d >= UPPER_BAND:
        return 1.0
    return (d - LOWER_BAND) / (UPPER_BAND - LOWER_BAND)


def build_market_data():
    """
    Fetch BTC + top-N alts daily data into a single DataFrame:
    columns: date, coin_id, price, mcap
    """
    print("Fetching top alts (excluding stables & BTC)...")
    alt_ids = get_top_alts_ex_stables(TOP_N_ALTS)
    print(f"Top {len(alt_ids)} alts:", alt_ids)

    all_ids = ["bitcoin"] + alt_ids

    effective_start, effective_end = get_effective_dates()
    from_ts = date_to_unix(effective_start.strftime("%Y-%m-%d"))
    to_ts = date_to_unix(END_DATE)

    frames = []
    for cid in all_ids:
        print(f"Fetching history for {cid}...")
        df_coin = get_market_chart_range(cid, "usd", from_ts, to_ts)
        frames.append(df_coin)

    df = pd.concat(frames, ignore_index=True)
    return df, alt_ids


def pivot_market_data(df):
    """
    Pivot the long-form data into wide daily tables:
    - price_df: index=date, columns=coin_id, values=price
    - mcap_df: index=date, columns=coin_id, values=mcap
    """
    price_df = df.pivot(index="date", columns="coin_id", values="price")
    mcap_df = df.pivot(index="date", columns="coin_id", values="mcap")

    price_df = price_df.sort_index()
    mcap_df = mcap_df.sort_index()

    return price_df, mcap_df


def run_backtest(price_df, mcap_df, alt_ids):
    """
    Run daily dominance rotation vs BTC-only DCA.
    Returns a DataFrame 'res' with:
      date, portfolio, btc_only, dom, wBTC
    """
    effective_start, effective_end = get_effective_dates()
    mask = (price_df.index >= effective_start) & (price_df.index <= effective_end)

    prices = price_df[mask].copy()
    mcaps = mcap_df[mask].copy()

    dates = prices.index.tolist()

    btc_col = "bitcoin"

    alt_mcaps = mcaps[alt_ids]
    total_alt_mcap = alt_mcaps.sum(axis=1)

    btc_mcap = mcaps[btc_col]
    dom = btc_mcap / (btc_mcap + total_alt_mcap)

    # initial state
    btc_units = 0.0
    alt_units = {cid: 0.0 for cid in alt_ids}

    # initial allocation on first day
    d0 = dom.iloc[0]
    w0 = w_btc_from_dom(d0)

    p0_btc = prices.loc[dates[0], btc_col]
    # alt weights proportional to their alt mcap on day 0
    alt_m0 = alt_mcaps.loc[dates[0]]
    alt_weight0 = alt_m0 / alt_m0.sum()

    init_btc_val = INITIAL_CAPITAL * w0
    init_alt_val = INITIAL_CAPITAL * (1 - w0)

    btc_units += init_btc_val / p0_btc
    for cid in alt_ids:
        price_ci = prices.loc[dates[0], cid]
        alloc_ci = init_alt_val * alt_weight0[cid]
        alt_units[cid] += alloc_ci / price_ci

    portfolio_values = []
    btc_only_values = []
    wbtc_series = []
    dom_series = []

    # BTC-only DCA benchmark
    btc_units_bench = INITIAL_CAPITAL / p0_btc

    for i, date in enumerate(dates):
        if i > 0:
            d = dom.loc[date]
            w = w_btc_from_dom(d)

            p_btc = prices.loc[date, btc_col]
            alt_m_t = alt_mcaps.loc[date]
            alt_weight_t = alt_m_t / alt_m_t.sum()

            # DCA flows
            buy_btc_val = DAILY_DCA * w
            buy_alts_val = DAILY_DCA * (1 - w)

            # Buy BTC
            btc_units += buy_btc_val / p_btc

            # Buy alts according to alt weights
            for cid in alt_ids:
                p_ci = prices.loc[date, cid]
                alloc_ci = buy_alts_val * alt_weight_t[cid]
                alt_units[cid] += alloc_ci / p_ci

            # Benchmark: all into BTC
            btc_units_bench += DAILY_DCA / p_btc
        else:
            d = d0
            w = w0

        # Compute portfolio value
        p_btc = prices.loc[date, btc_col]
        cur_btc_val = btc_units * p_btc

        cur_alt_val = 0.0
        for cid in alt_ids:
            p_ci = prices.loc[date, cid]
            cur_alt_val += alt_units[cid] * p_ci

        pv = cur_btc_val + cur_alt_val

        # Rebalance to target
        target_btc_val = pv * w
        delta_btc_val = target_btc_val - cur_btc_val

        if abs(pv) < 1e-9:
            pass
        elif delta_btc_val > 0 and cur_alt_val > 0:
            # sell alts proportionally to buy BTC
            sell_frac = min(delta_btc_val / cur_alt_val, 1.0)
            value_sold = 0.0
            for cid in alt_ids:
                p_ci = prices.loc[date, cid]
                val_ci = alt_units[cid] * p_ci
                sell_ci_val = val_ci * sell_frac
                alt_units[cid] -= sell_ci_val / p_ci
                value_sold += sell_ci_val
            btc_units += value_sold / p_btc

        elif delta_btc_val < 0:
            # sell BTC, buy alts by alt weights
            sell_val = -delta_btc_val
            sell_val = min(sell_val, cur_btc_val)
            btc_units -= sell_val / p_btc

            alt_m_t = alt_mcaps.loc[date]
            alt_weight_t = alt_m_t / alt_m_t.sum()

            for cid in alt_ids:
                p_ci = prices.loc[date, cid]
                buy_ci_val = sell_val * alt_weight_t[cid]
                alt_units[cid] += buy_ci_val / p_ci

        # re-evaluate portfolio after rebalance
        cur_btc_val = btc_units * prices.loc[date, btc_col]
        cur_alt_val = sum(
            alt_units[cid] * prices.loc[date, cid] for cid in alt_ids
        )
        pv = cur_btc_val + cur_alt_val

        portfolio_values.append(pv)
        wbtc_series.append(w)
        dom_series.append(d)

        # benchmark value
        btc_only_values.append(btc_units_bench * prices.loc[date, btc_col])

    res = pd.DataFrame(
        {
            "date": dates,
            "portfolio": portfolio_values,
            "btc_only": btc_only_values,
            "wBTC": wbtc_series,
            "dominance": dom_series,
        }
    )

    return res


def summarize_results(res):
    """
    Print annual and total performance summary and save CSVs.
    """
    res["date"] = pd.to_datetime(res["date"])
    res.set_index("date", inplace=True)

    annual = res[["portfolio", "btc_only"]].resample("A").last()
    annual.index = annual.index.year

    print("\n=== Annual Results (end-of-year values) ===")
    print(annual)

    print("\n=== Total Period ===")
    start_port = res["portfolio"].iloc[0]
    end_port = res["portfolio"].iloc[-1]
    start_bench = res["btc_only"].iloc[0]
    end_bench = res["btc_only"].iloc[-1]

    print(f"Strategy: {start_port:.2f} → {end_port:.2f}")
    print(f"BTC-only: {start_bench:.2f} → {end_bench:.2f}")

    res.to_csv(os.path.join(OUT_DIR, "daily_results.csv"))
    annual.to_csv(os.path.join(OUT_DIR, "annual_summary.csv"))
    print(f"\nSaved daily_results.csv and annual_summary.csv in {OUT_DIR}/")


def main():
    print("=== BTC Dominance Rotation Backtest (BTC vs Top-20 Alts, ex-stables) ===")
    eff_start, eff_end = get_effective_dates()
    print(f"Desired date range: {DESIRED_START_DATE} → {END_DATE}")
    print(f"Effective date range (API-limited): {eff_start} → {eff_end}")
    print(f"Dominance bands: {LOWER_BAND:.2f} – {UPPER_BAND:.2f}")
    print("NOTE: These bands are provisional; we should revisit & widen them later.\n")

    df_raw, alt_ids = build_market_data()
    price_df, mcap_df = pivot_market_data(df_raw)

    print("\nRunning backtest...")
    res = run_backtest(price_df, mcap_df, alt_ids)
    summarize_results(res)


if __name__ == "__main__":
    main()
    
