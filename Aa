"""
BTC Dominance Rotation Backtest (CoinMetrics version)
-----------------------------------------------------

Data source:
- CoinMetrics Community API (no API key required).

We fetch daily:
    - PriceUSD
    - CapMrktCurUSD  (current market cap in USD)

For assets:
    - btc
    - eth
    - sol
    - bnb

Dominance (daily):
    dom(t) = CapBTC(t) / (CapBTC(t) + CapETH(t) + CapSOL(t) + CapBNB(t))

Allocation rule (your linear 75–81% scheme):

- Dominance range: 0.75 -> 0.81
- Midpoint: 0.78
- Mapping:

    dom <= 0.75  => 100% BTC, 0% alts
    dom = 0.76   => 83.33% BTC, 16.67% alts
    dom = 0.77   => 66.67% BTC, 33.33% alts
    dom = 0.78   => 50% BTC, 50% alts
    dom = 0.79   => 33.33% BTC, 66.67% alts
    dom = 0.80   => 16.67% BTC, 83.33% alts
    dom >= 0.81  => 0% BTC, 100% alts

i.e.

    w_alts = (dom - 0.75) / 0.06, clamped to [0, 1]
    w_btc  = 1 - w_alts

Strategy:
- Initial capital: $100 on first day.
- Daily DCA: $10 per day.
- Each day:
    - Compute dominance.
    - Compute target weights (wBTC, wAlt).
    - Allocate DAILY_DCA across BTC / alts by those weights.
    - Rebalance BTC vs alt basket to target weights.
- Alt basket: market-cap–weighted mix of ETH, SOL, BNB.
- Benchmark: BTC-only DCA with the same flows.
"""

import os
import time
from datetime import datetime

import requests
import pandas as pd

# ---------------- CONFIG ----------------

BASE_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"

ASSETS = ["btc", "eth", "sol", "bnb"]
METRICS = ["PriceUSD", "CapMrktCurUSD"]

START_DATE = "2023-01-01"
END_DATE   = "2025-11-01"

# Dominance band
DOM_LOWER = 0.75
DOM_UPPER = 0.81

INITIAL_CAPITAL = 100.0
DAILY_DCA       = 10.0

OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------- DATA FETCH ----------------

def cm_get(asset, start_date, end_date, metrics):
    """
    Fetch daily asset metrics from CoinMetrics community API.
    """
    params = {
        "assets": asset,
        "metrics": ",".join(metrics),
        "frequency": "1d",
        "start_time": start_date,
        "end_time": end_date,
    }
    resp = requests.get(BASE_URL, params=params)
    if resp.status_code != 200:
        raise RuntimeError(f"CoinMetrics error {resp.status_code}: {resp.text[:300]}")

    js = resp.json()
    rows = js.get("data", [])
    if not rows:
        raise RuntimeError(f"No data returned for asset {asset}")

    records = []
    for row in rows:
        ts_str = row["time"][:10]  # 'YYYY-MM-DD'
        dt = datetime.strptime(ts_str, "%Y-%m-%d").date()
        rec = {"date": dt}
        for m in metrics:
            val_str = row.get(m)
            val = float(val_str) if val_str is not None else None
            rec[m] = val
        records.append(rec)

    df = pd.DataFrame(records)
    return df


def build_market_data():
    """
    Fetch BTC + ETH + SOL + BNB daily PriceUSD + CapMrktCurUSD.
    """
    frames = []
    for asset in ASSETS:
        print(f"Fetching CoinMetrics data for {asset} …")
        df_asset = cm_get(asset, START_DATE, END_DATE, METRICS)
        df_asset["asset"] = asset
        frames.append(df_asset)
        time.sleep(0.5)  # small pause to be nice

    df = pd.concat(frames, ignore_index=True)
    return df


def pivot_market_data(df):
    """
    Produce two DataFrames:
    - prices: rows = date, columns = asset, values = PriceUSD
    - caps  : rows = date, columns = asset, values = CapMrktCurUSD
    """
    prices = df.pivot(index="date", columns="asset", values="PriceUSD")
    caps   = df.pivot(index="date", columns="asset", values="CapMrktCurUSD")

    prices = prices.sort_index()
    caps   = caps.sort_index()

    return prices, caps


# ---------------- WEIGHT FUNCTION ----------------

def w_btc_from_dom(dom):
    """
    Linear mapping between DOM_LOWER (0.75) and DOM_UPPER (0.81):

      dom <= 0.75  -> 100% BTC
      dom >= 0.81  -> 100% alts

      0.75 < dom < 0.81:
          w_alts = (dom - 0.75) / 0.06
          w_btc  = 1 - w_alts
    """
    if dom <= DOM_LOWER:
        return 1.0, 0.0
    if dom >= DOM_UPPER:
        return 0.0, 1.0

    w_alts = (dom - DOM_LOWER) / (DOM_UPPER - DOM_LOWER)
    if w_alts < 0:
        w_alts = 0.0
    if w_alts > 1:
        w_alts = 1.0

    w_btc = 1.0 - w_alts
    return w_btc, w_alts


# ---------------- BACKTEST ENGINE ----------------

def run_backtest(prices, caps):
    dates = prices.index.tolist()

    btc_col = "btc"
    alt_assets = [a for a in ASSETS if a != "btc"]

    # Alt market caps and total
    alt_caps = caps[alt_assets]
    total_alt_caps = alt_caps.sum(axis=1)

    btc_caps = caps[btc_col]
    dominance = btc_caps / (btc_caps + total_alt_caps)

    # --- Initial allocation ---
    btc_units = 0.0
    alt_units = {a: 0.0 for a in alt_assets}

    d0 = dominance.iloc[0]
    wBTC0, wAlt0 = w_btc_from_dom(d0)

    p0_btc = prices.loc[dates[0], btc_col]
    alt_caps_0 = alt_caps.loc[dates[0]]
    alt_weights_0 = alt_caps_0 / alt_caps_0.sum()

    init_btc_val = INITIAL_CAPITAL * wBTC0
    init_alt_val = INITIAL_CAPITAL * wAlt0

    btc_units += init_btc_val / p0_btc

    for a in alt_assets:
        p_a0 = prices.loc[dates[0], a]
        alloc_a0 = init_alt_val * alt_weights_0[a]
        alt_units[a] += alloc_a0 / p_a0

    # BTC-only benchmark
    btc_units_bench = INITIAL_CAPITAL / p0_btc

    portfolio_values = []
    btc_only_values  = []
    wbtc_series      = []
    dom_series       = []

    for i, date in enumerate(dates):
        d = dominance.loc[date]
        wBTC, wAlt = w_btc_from_dom(d)

        p_btc = prices.loc[date, btc_col]
        alt_caps_t = alt_caps.loc[date]
        alt_weights_t = alt_caps_t / alt_caps_t.sum()

        if i > 0:
            # DCA flows
            buy_btc_val  = DAILY_DCA * wBTC
            buy_alts_val = DAILY_DCA * wAlt

            # Buy BTC
            btc_units += buy_btc_val / p_btc

            # Buy alts in proportion to alt weights
            for a in alt_assets:
                p_a = prices.loc[date, a]
                alloc_a = buy_alts_val * alt_weights_t[a]
                alt_units[a] += alloc_a / p_a

            # Benchmark DCA: all BTC
            btc_units_bench += DAILY_DCA / p_btc

        # Portfolio value before rebalance
        cur_btc_val = btc_units * p_btc
        cur_alt_val = sum(alt_units[a] * prices.loc[date, a] for a in alt_assets)
        pv = cur_btc_val + cur_alt_val

        # Rebalance to target
        target_btc_val = pv * wBTC
        delta = target_btc_val - cur_btc_val

        if delta > 0 and cur_alt_val > 0:
            # Need more BTC => sell alts proportionally
            sell_frac = min(delta / cur_alt_val, 1.0)
            value_sold = 0.0
            for a in alt_assets:
                p_a = prices.loc[date, a]
                val_a = alt_units[a] * p_a
                sell_a_val = val_a * sell_frac
                alt_units[a] -= sell_a_val / p_a
                value_sold += sell_a_val
            btc_units += value_sold / p_btc

        elif delta < 0:
            # Need more alts => sell BTC, buy alts
            sell_val = min(-delta, cur_btc_val)
            btc_units -= sell_val / p_btc

            alt_caps_t = alt_caps.loc[date]
            alt_weights_t = alt_caps_t / alt_caps_t.sum()

            for a in alt_assets:
                p_a = prices.loc[date, a]
                buy_val_a = sell_val * alt_weights_t[a]
                alt_units[a] += buy_val_a / p_a

        # Recompute value after rebalance
        cur_btc_val = btc_units * prices.loc[date, btc_col]
        cur_alt_val = sum(alt_units[a] * prices.loc[date, a] for a in alt_assets)
        pv = cur_btc_val + cur_alt_val

        portfolio_values.append(pv)
        wbtc_series.append(wBTC)
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


# ---------------- SUMMARY ----------------

def summarize_results(res):
    res["date"] = pd.to_datetime(res["date"])
    res.set_index("date", inplace=True)

    annual = res[["portfolio", "btc_only"]].resample("YE").last()
    annual.index = annual.index.year

    print("\n=== ANNUAL SUMMARY ===")
    print(annual)

    print("\n=== TOTAL PERIOD ===")
    print("Strategy: ", res["portfolio"].iloc[0], "→", res["portfolio"].iloc[-1])
    print("BTC-only:", res["btc_only"].iloc[0],   "→", res["btc_only"].iloc[-1])

    os.makedirs(OUT_DIR, exist_ok=True)
    res.to_csv(f"{OUT_DIR}/daily_results.csv", index=True)
    annual.to_csv(f"{OUT_DIR}/annual_summary.csv", index=True)
    print(f"\nSaved CSVs in {OUT_DIR}/")


# ---------------- MAIN ----------------

def main():
    print("=== BTC Dominance Rotation Backtest (BTC vs ETH+SOL+BNB, CoinMetrics) ===")
    print(f"Date range: {START_DATE} → {END_DATE}")
    print(f"Dominance band: {DOM_LOWER} – {DOM_UPPER} (midpoint {(DOM_LOWER+DOM_UPPER)/2:.2f})\n")

    df_raw = build_market_data()
    prices, caps = pivot_market_data(df_raw)

    print("\nRunning backtest…")
    res = run_backtest(prices, caps)
    summarize_results(res)


if __name__ == "__main__":
    main()
