"""
BTC Dominance Rotation Backtest
-------------------------------

Universe:
- BTC + fixed alt basket: ETH, SOL, BNB.

Dominance signal (daily):
    dom(t) = BTC_mcap(t) / (BTC_mcap(t) + mcap_ETH(t) + mcap_SOL(t) + mcap_BNB(t))

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

    w_alts = (dom - 0.75) / 0.06,   clamped to [0, 1]
    w_btc  = 1 - w_alts

DCA + rebalancing:
- Start with $100 on the first day, split by wBTC(dom(start)).
- Each day, add $10:
    - Allocate DAILY_DCA using today's wBTC / wAlt.
- Then rebalance BTC vs alt basket to target weights.
- Alts are held as a market-cap–weighted basket across ETH, SOL, BNB.
- Benchmark: same cash-flow, but 100% BTC-only DCA.
"""

import os
import time
from datetime import datetime, timezone, timedelta

import requests
import pandas as pd

# ---------------- CONFIG ----------------

COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")

if not COINGECKO_API_KEY:
    raise RuntimeError(
        "COINGECKO_API_KEY environment variable is not set. "
        "Set it as a GitHub Secret: COINGECKO_API_KEY"
    )

BASE_URL = "https://api.coingecko.com/api/v3"

# Desired overall period (will be clamped by 365-day API limit)
DESIRED_START_DATE = "2023-01-01"
END_DATE           = "2025-11-01"

# Fixed alt list
ALT_IDS_FIXED = [
    "ethereum",
    "solana",
    "binancecoin",
]

# Dominance band
DOM_LOWER = 0.75
DOM_UPPER = 0.81

INITIAL_CAPITAL = 100.0
DAILY_DCA       = 10.0

OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------- API HELPERS ----------------

def cg_get(path, params=None, sleep_sec=1.2):
    """GET request to CoinGecko using API key with crude rate limiting."""
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
    mcaps  = pd.DataFrame(js["market_caps"], columns=["ts", "mcap"])

    prices["date"] = pd.to_datetime(prices["ts"], unit="ms").dt.date
    mcaps["date"]  = pd.to_datetime(mcaps["ts"],  unit="ms").dt.date

    prices = prices.groupby("date")["price"].last().reset_index()
    mcaps  = mcaps.groupby("date")["mcap"].last().reset_index()

    df = pd.merge(prices, mcaps, on="date", how="outer").sort_values("date")
    df["coin_id"] = coin_id
    return df


def date_to_unix(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return dt.timestamp()


def get_effective_dates():
    """
    CoinGecko free plan: only last ~365 days from today.
    So:
    - Effective end date = min(END_DATE, today)
    - Effective start date = max(DESIRED_START_DATE, today - 365 days)
    """
    desired_start = datetime.strptime(DESIRED_START_DATE, "%Y-%m-%d").date()
    desired_end   = datetime.strptime(END_DATE, "%Y-%m-%d").date()

    today = datetime.utcnow().date()
    effective_end = min(desired_end, today)

    earliest_allowed = today - timedelta(days=365)
    effective_start = max(desired_start, earliest_allowed)

    return effective_start, effective_end


# ---------------- WEIGHT FUNCTION ----------------

def w_btc_from_dom(dom):
    """
    Linear mapping between 0.75 and 0.81:
      dom <= 0.75  -> 100% BTC
      dom >= 0.81  -> 100% alts
      in between   -> interpolate linearly with midpoint 0.78

    w_alts = (dom - 0.75) / 0.06, clamped [0,1]
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


# ---------------- DATA BUILD ----------------

def build_market_data():
    """
    Fetch BTC + ETH + SOL + BNB daily data.
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


# ---------------- BACKTEST ----------------

def run_backtest(price_df, mcap_df, alt_ids):
    effective_start, effective_end = get_effective_dates()
    mask = (price_df.index >= effective_start) & (price_df.index <= effective_end)

    prices = price_df[mask].copy()
    mcaps  = mcap_df[mask].copy()

    dates = prices.index.tolist()

    btc_col = "bitcoin"
    alt_mcaps       = mcaps[alt_ids]
    total_alt_mcap  = alt_mcaps.sum(axis=1)
    btc_mcap        = mcaps[btc_col]
    dom             = btc_mcap / (btc_mcap + total_alt_mcap)

    # --- Initial allocation ---
    btc_units = 0.0
    alt_units = {cid: 0.0 for cid in alt_ids}

    d0 = dom.iloc[0]
    wBTC0, wAlt0 = w_btc_from_dom(d0)

    p0_btc = prices.loc[dates[0], btc_col]
    alt_m0 = alt_mcaps.loc[dates[0]]
    alt_weight0 = alt_m0 / alt_m0.sum()

    init_btc_val = INITIAL_CAPITAL * wBTC0
    init_alt_val = INITIAL_CAPITAL * wAlt0

    btc_units += init_btc_val / p0_btc

    for cid in alt_ids:
        p_ci = prices.loc[dates[0], cid]
        alloc_ci = init_alt_val * alt_weight0[cid]
        alt_units[cid] += alloc_ci / p_ci

    portfolio_values = []
    btc_only_values  = []
    wbtc_series      = []
    dom_series       = []

    # BTC-only DCA benchmark units
    btc_units_bench = INITIAL_CAPITAL / p0_btc

    for i, date in enumerate(dates):
        d = dom.loc[date]
        wBTC, wAlt = w_btc_from_dom(d)

        p_btc = prices.loc[date, btc_col]
        alt_m_t = alt_mcaps.loc[date]
        alt_weight_t = alt_m_t / alt_m_t.sum()

        if i > 0:
            # Apply DCA flows
            buy_btc_val  = DAILY_DCA * wBTC
            buy_alts_val = DAILY_DCA * wAlt

            # Buy BTC
            btc_units += buy_btc_val / p_btc

            # Buy alts
            for cid in alt_ids:
                p_ci = prices.loc[date, cid]
                alloc_ci = buy_alts_val * alt_weight_t[cid]
                alt_units[cid] += alloc_ci / p_ci

            # Benchmark DCA: all into BTC
            btc_units_bench += DAILY_DCA / p_btc

        # Portfolio value before rebalance
        cur_btc_val = btc_units * p_btc
        cur_alt_val = sum(alt_units[cid] * prices.loc[date, cid] for cid in alt_ids)
        pv = cur_btc_val + cur_alt_val

        # Rebalance to target weights
        target_btc_val = pv * wBTC
        delta = target_btc_val - cur_btc_val

        if delta > 0 and cur_alt_val > 0:
            # Sell alts proportionally, buy BTC
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
            # Sell BTC, buy alts by market-cap weights
            sell_val = min(-delta, cur_btc_val)
            btc_units -= sell_val / p_btc

            alt_m_t = alt_mcaps.loc[date]
            alt_weight_t = alt_m_t / alt_m_t.sum()

            for cid in alt_ids:
                p_ci = prices.loc[date, cid]
                buy_val_ci = sell_val * alt_weight_t[cid]
                alt_units[cid] += buy_val_ci / p_ci

        # Recompute value after rebalance
        cur_btc_val = btc_units * prices.loc[date, btc_col]
        cur_alt_val = sum(alt_units[cid] * prices.loc[date, cid] for cid in alt_ids)
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

    # Annual summary (year-end)
    annual = res[["portfolio", "btc_only"]].resample("YE").last()
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


# ---------------- MAIN ----------------

def main():
    print("=== BTC Dominance Rotation Backtest (BTC vs ETH+SOL+BNB) ===")
    eff_start, eff_end = get_effective_dates()
    print(f"Desired:  {DESIRED_START_DATE} → {END_DATE}")
    print(f"Effective (API-limited): {eff_start} → {eff_end}")
    print(f"Dominance band: {DOM_LOWER} – {DOM_UPPER} (midpoint {0.5*(DOM_LOWER+DOM_UPPER):.2f})\n")

    df_raw, alt_ids = build_market_data()
    price_df, mcap_df = pivot_market_data(df_raw)

    print("\nRunning backtest…")
    res = run_backtest(price_df, mcap_df, alt_ids)
    summarize_results(res)


if __name__ == "__main__":
    main()

