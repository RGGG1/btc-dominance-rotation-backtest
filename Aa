"""
BTC Dominance Rotation Backtest
-------------------------------

Strategy:
- Universe: BTC + ETH + SOL + BNB (fixed alt list).
- Dominance signal (daily):
    dom(t) = BTC_mcap(t) / (BTC_mcap(t) + alt_mcap_total(t))

- Bands (you can tune):
    lower_band = 0.60
    upper_band = 0.75

- BTC weight:
    if dom <= LOWER_BAND: wBTC = 0 (all alts)
    if dom >= UPPER_BAND: wBTC = 1 (all BTC)
    else: linearly interpolate

- Alts bucket: market-cap–weighted basket of ETH, SOL, BNB.
- Start: DESIRED_START_DATE (but actual start may be clamped by API limits)
    - Allocate initial $100 according to wBTC(start).
- Daily:
    - Add $10 DCA
    - Allocate by wBTC
    - Rebalance BTC vs alt basket
- Benchmark: BTC-only DCA
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
        "Set it as a GitHub Secret: COINGECKO_API_KEY"
    )

BASE_URL = "https://api.coingecko.com/api/v3"

# Desired period (but actual will be clamped by 365-day rule)
DESIRED_START_DATE = "2023-01-01"
END_DATE           = "2025-11-01"

# Fixed alt list (no survivorship bias)
ALT_IDS_FIXED = [
    "ethereum",
    "solana",
    "binancecoin",
]

LOWER_BAND = 0.60
UPPER_BAND = 0.75

INITIAL_CAPITAL = 100.0
DAILY_DCA = 10.0

OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------- API HELPERS ----------------

def cg_get(path, params=None, sleep_sec=1.2):
    """
    GET request to CoinGecko using API key.
    Handles crude rate limiting.
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

    time.sleep(sleep_sec)
    return resp.json()


def get_market_chart_range(coin_id, vs_currency, from_ts, to_ts):
    """
    Fetch historical prices + market caps between unix timestamps.
    Returns daily dataframe for one coin.
    """
    js = cg_get(
        f"/coins/{coin_id}/market_chart/range",
        params={
            "vs_currency": vs_currency,
            "from": int(from_ts),
            "to": int(to_ts),
        },
    )

    prices = pd.DataFrame(js["prices"], columns=["ts", "price"])
    mcaps = pd.DataFrame(js["market_caps"], columns=["ts", "mcap"])

    prices["date"] = pd.to_datetime(prices["ts"], unit="ms").dt.date
    mcaps["date"] = pd.to_datetime(mcaps["ts"], unit="ms").dt.date

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
    CoinGecko free plans allow ONLY the last 365 days *from today*.
    So:
    - End date = min(END_DATE, today)
    - Start date = max(DESIRED_START_DATE, today - 365 days)
    """
    desired_start = datetime.strptime(DESIRED_START_DATE, "%Y-%m-%d").date()
    desired_end   = datetime.strptime(END_DATE, "%Y-%m-%d").date()

    today = datetime.utcnow().date()
    effective_end = min(desired_end, today)

    earliest_allowed = today - timedelta(days=365)
    effective_start = max(desired_start, earliest_allowed)

    return effective_start, effective_end


# ---------------- STRATEGY ----------------

def w_btc_from_dom(d):
    if d <= LOWER_BAND:
        return 0.0
    if d >= UPPER_BAND:
        return 1.0
    return (d - LOWER_BAND) / (UPPER_BAND - LOWER_BAND)


def build_market_data():
    """
    Fetch BTC + ETH + SOL + BNB market data.
    """
    alt_ids = ALT_IDS_FIXED
    print("Using fixed alt list:", alt_ids)

    all_ids = ["bitcoin"] + alt_ids

    effective_start, effective_end = get_effective_dates()
    from_ts = date_to_unix(effective_start.strftime("%Y-%m-%d"))
    to_ts   = date_to_unix(effective_end.strftime("%Y-%m-%d"))

    frames = []
    for cid in all_ids:
        print(f"Fetching history for {cid} …")
        df_coin = get_market_chart_range(cid, "usd", from_ts, to_ts)
        frames.append(df_coin)

    df = pd.concat(frames, ignore_index=True)
    return df, alt_ids


def pivot_market_data(df):
    price_df = df.pivot(index="date", columns="coin_id", values="price")
    mcap_df  = df.pivot(index="date", columns="coin_id", values="mcap")
    price_df = price_df.sort_index()
    mcap_df  = mcap_df.sort_index()
    return price_df, mcap_df


def run_backtest(price_df, mcap_df, alt_ids):
    effective_start, effective_end = get_effective_dates()
    mask = (price_df.index >= effective_start) & (price_df.index <= effective_end)

    prices = price_df[mask].copy()
    mcaps  = mcap_df[mask].copy()

    dates = prices.index.tolist()

    btc_col = "bitcoin"
    alt_mcaps = mcaps[alt_ids]
    total_alt_mcap = alt_mcaps.sum(axis=1)

    btc_mcap = mcaps[btc_col]
    dom = btc_mcap / (btc_mcap + total_alt_mcap)

    # --- Initial allocation ---
    btc_units = 0.0
    alt_units = {cid: 0.0 for cid in alt_ids}

    d0 = dom.iloc[0]
    w0 = w_btc_from_dom(d0)

    p0_btc = prices.loc[dates[0], btc_col]
    alt_m0 = alt_mcaps.loc[dates[0]]
    alt_weight0 = alt_m0 / alt_m0.sum()

    init_btc_val = INITIAL_CAPITAL * w0
    init_alt_val = INITIAL_CAPITAL * (1 - w0)

    btc_units += init_btc_val / p0_btc

    for cid in alt_ids:
        p_ci = prices.loc[dates[0], cid]
        alloc_ci = init_alt_val * alt_weight0[cid]
        alt_units[cid] += alloc_ci / p_ci

    # --- Trackers ---
    portfolio_values = []
    btc_only_values = []
    wbtc_series = []
    dom_series = []

    btc_units_bench = INITIAL_CAPITAL / p0_btc

    for i, date in enumerate(dates):
        if i > 0:
            d = dom.loc[date]
            w = w_btc_from_dom(d)

            p_btc = prices.loc[date, btc_col]
            alt_m_t = alt_mcaps.loc[date]
            alt_weight_t = alt_m_t / alt_m_t.sum()

            # DCA
            buy_btc_val  = DAILY_DCA * w
            buy_alts_val = DAILY_DCA * (1 - w)

            btc_units += buy_btc_val / p_btc

            for cid in alt_ids:
                p_ci = prices.loc[date, cid]
                alloc_ci = buy_alts_val * alt_weight_t[cid]
                alt_units[cid] += alloc_ci / p_ci

            # BTC-only DCA
            btc_units_bench += DAILY_DCA / p_btc

        else:
            d = d0
            w = w0

        # Value before rebalance
        p_btc = prices.loc[date, btc_col]
        cur_btc_val = btc_units * p_btc
        cur_alt_val = sum(alt_units[cid] * prices.loc[date, cid] for cid in alt_ids)

        pv = cur_btc_val + cur_alt_val

        target_btc_val = pv * w
        delta = target_btc_val - cur_btc_val

        # Rebalance
        if delta > 0 and cur_alt_val > 0:
            sell_frac = min(delta / cur_alt_val, 1.0)
            value_sold = 0.0
            for cid in alt_ids:
                p_ci = prices.loc[date, cid]
                val_ci = alt_units[cid] * p_ci
                sell_val_ci = val_ci * sell_frac
                alt_units[cid] -= sell_val_ci / p_ci
                value_sold += sell_val_ci
            btc_units += value_sold / p_btc

        elif delta < 0:
            sell_val = -delta
            sell_val = min(sell_val, cur_btc_val)
            btc_units -= sell_val / p_btc

            alt_m_t = alt_mcaps.loc[date]
            alt_weight_t = alt_m_mcaps = alt_mcaps.loc[date] / alt_mcaps.loc[date].sum()

            for cid in alt_ids:
                p_ci = prices.loc[date, cid]
                buy_val_ci = sell_val * alt_weight_t[cid]
                alt_units[cid] += buy_val_ci / p_ci

        cur_btc_val = btc_units * prices.loc[date, btc_col]
        cur_alt_val = sum(alt_units[cid] * prices.loc[date, cid] for cid in alt_ids)
        pv = cur_btc_val + cur_alt_val

        portfolio_values.append(pv)
        wbtc_series.append(w)
        dom_series.append(d)
        btc_only_values.append(btc_units_bench * prices.loc[date, btc_col])

    res = pd.DataFrame({
        "date": dates,
        "portfolio": portfolio_values,
        "btc_only": btc_only_values,
        "wBTC": wbtc_series,
        "dominance": dom_series,
    })
    return res


def summarize_results(res):
    res["date"] = pd.to_datetime(res["date"])
    res.set_index("date", inplace=True)

    annual = res[["portfolio","btc_only"]].resample("A").last()
    annual.index = annual.index.year

    print("\n=== ANNUAL SUMMARY ===")
    print(annual)

    print("\n=== TOTAL PERIOD ===")
    print("Strategy: ", res["portfolio"].iloc[0], "→", res["portfolio"].iloc[-1])
    print("BTC-only:", res["btc_only"].iloc[0],   "→", res["btc_only"].iloc[-1])

    os.makedirs(OUT_DIR, exist_ok=True)
    res.to_csv(f"{OUT_DIR}/daily_results.csv")
    annual.to_csv(f"{OUT_DIR}/annual_summary.csv")
    print(f"\nSaved CSVs in {OUT_DIR}/")


def main():
    print("=== BTC Dominance Rotation Backtest (BTC vs ETH+SOL+BNB) ===")
    eff_start, eff_end = get_effective_dates()
    print(f"Desired:  {DESIRED_START_DATE} → {END_DATE}")
    print(f"Effective (API-limited): {eff_start} → {eff_end}")
    print(f"Dominance bands: {LOWER_BAND} – {UPPER_BAND}\n")

    df_raw, alt_ids = build_market_data()
    price_df, mcap_df = pivot_market_data(df_raw)

    print("\nRunning backtest…")
    res = run_backtest(price_df, mcap_df, alt_ids)
    summarize_results(res)


if __name__ == "__main__":
    main()
