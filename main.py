import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import requests

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"} 
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    tables = pd.read_html(response.text) 
    df = tables[0]
    tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
    return tickers


def download_tickers_tidy(tickers, d):
    start_date = datetime.now() - timedelta(days=d)
    end_date = datetime.now()
    
    df = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        group_by='ticker',
        auto_adjust=False
    )
    
    df = df.stack(level=0).reset_index()
    df.columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    
    return df

sp500_tickers = get_sp500_tickers()
print("First 10 tickers:", sp500_tickers[:10])

data_tidy = download_tickers_tidy(sp500_tickers[:10], 365)
print(data_tidy.head())


import numpy as np
import pandas as pd

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up / down
    return 100 - (100 / (1 + rs))

def true_range(high, low, close):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def zscore(s, window=60, min_periods=20):
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std()
    return (s - mu) / sd

def compute_action_table(data_tidy):
    df = data_tidy.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Ticker', 'Date'])

    feat = []
    for tkr, g in df.groupby('Ticker', sort=False):
        g = g.copy()
        g['ret1'] = g['Close'].pct_change(1)
        g['ret5'] = g['Close'].pct_change(5)
        g['ret20'] = g['Close'].pct_change(20)

        g['gap_ovr'] = (g['Open'] / g['Close'].shift(1) - 1)

        g['TR'] = true_range(g['High'], g['Low'], g['Close'])
        g['ATR14'] = g['TR'].rolling(14, min_periods=5).mean()
        g['ATRpct'] = g['ATR14'] / g['Close']

        g['vol20'] = g['ret1'].rolling(20, min_periods=10).std() * np.sqrt(252)

        g['vol_z'] = zscore(g['Volume'].astype(float), window=60, min_periods=20)

        g['RSI14'] = rsi(g['Close'], period=14)

        rolling_min = g['Close'].rolling(252, min_periods=60).min()
        rolling_max = g['Close'].rolling(252, min_periods=60).max()
        g['pct_52w'] = (g['Close'] - rolling_min) / (rolling_max - rolling_min)

        g['atr_z'] = zscore(g['ATRpct'], window=60, min_periods=20)
        g['abs_ret1'] = g['ret1'].abs()
        g['abs_ret1_z'] = zscore(g['abs_ret1'], window=60, min_periods=20)
        g['gap_abs'] = g['gap_ovr'].abs()
        g['gap_z'] = zscore(g['gap_abs'], window=60, min_periods=20)

        g['ActionScore'] = (
            0.4 * g['vol_z'].abs() +
            0.3 * g['atr_z'].abs() +
            0.2 * g['abs_ret1_z'].abs() +
            0.1 * g['gap_z'].abs()
        )


        bias = []
        for i in range(len(g)):
            rsi_now = g['RSI14'].iloc[i]
            r5 = g['ret5'].iloc[i]
            r20 = g['ret20'].iloc[i]
            gap = g['gap_ovr'].iloc[i]

            b = "watch"

            if r5 > 0 and r20 > 0:
                b = "buy"
            elif r5 < 0 and r20 < 0:
                b = "sell"

            if pd.notna(rsi_now):
                if rsi_now >= 75:
                    b = "sell"
                elif rsi_now <= 25:
                    b = "buy"

            if b == "watch" and pd.notna(gap):
                if gap > 0.01:
                    b = "buy"
                elif gap < -0.01:
                    b = "sell"

            bias.append(b)
        g['Bias'] = bias

        feat.append(g)

    feats = pd.concat(feat, ignore_index=True)

    latest_date = feats['Date'].max()
    snap = feats.loc[feats['Date'] == latest_date, [
        'Ticker', 'Date', 'ActionScore', 'Bias',
        'ret1', 'ret5', 'ret20', 'gap_ovr',
        'ATRpct', 'vol20', 'vol_z', 'RSI14', 'pct_52w'
    ]].copy()

    snap['Rank'] = snap['ActionScore'].rank(ascending=False, method='first')
    snap = snap.sort_values('ActionScore', ascending=False)

    def pct(x): 
        return (x*100).round(2)

    out = snap.copy()
    out.rename(columns={
        'ret1':'Ret1d_%', 'ret5':'Ret5d_%', 'ret20':'Ret20d_%',
        'gap_ovr':'GapOvernight_%', 'ATRpct':'ATR_%', 'vol20':'Vol20_Ann', 
        'vol_z':'VolZ', 'pct_52w':'Pct_52w', 'RSI14':'RSI14'
    }, inplace=True)

    for c in ['Ret1d_%','Ret5d_%','Ret20d_%','GapOvernight_%','ATR_%','Pct_52w']:
        out[c] = pct(out[c])

    out['Vol20_Ann'] = (out['Vol20_Ann']*100).round(2)
    out['ActionScore'] = out['ActionScore'].round(2)
    out['VolZ'] = out['VolZ'].round(2)
    out['RSI14'] = out['RSI14'].round(1)

    cols = ['Rank','Ticker','Bias','ActionScore','Ret1d_%','Ret5d_%','Ret20d_%',
            'GapOvernight_%','ATR_%','Vol20_Ann','VolZ','RSI14','Pct_52w','Date']
    out = out[cols].sort_values('Rank').reset_index(drop=True)

    return out, latest_date

action_table, snap_date = compute_action_table(data_tidy)
print(f"Action snapshot for {snap_date.date()}:")
print(action_table.head(10).to_string(index=False))

def make_weekly_plan(action_table, lookahead=5):
    plan = action_table.copy()

    if 'Close' not in plan.columns:
        raise ValueError("Need latest Close in snapshot to compute targets/stops")

    # Target: bias * expected return (use 5d return as proxy for weekly move)
    exp_move = plan['Ret5d_%'] / 100.0  # convert back to ratio

    plan['Entry'] = plan['Close']

    plan['Target'] = np.where(
        plan['Bias'] == "buy",
        plan['Entry'] * (1 + exp_move.abs()),  # upside move
        np.where(plan['Bias'] == "sell",
                 plan['Entry'] * (1 - exp_move.abs()),  # downside move
                 plan['Entry'])  # neutral = hold flat
    )

    atr = plan['ATR_%'] / 100.0
    plan['Stop'] = np.where(
        plan['Bias'] == "buy",
        plan['Entry'] * (1 - 2 * atr),   # 2× ATR buffer
        np.where(plan['Bias'] == "sell",
                 plan['Entry'] * (1 + 2 * atr),
                 plan['Entry'])
    )

    # Conviction: scale of 1–5 based on ActionScore rank percentile
    rank_pct = plan['Rank'] / len(plan)
    plan['Conviction'] = np.select(
        [
            rank_pct <= 0.1,
            rank_pct <= 0.25,
            rank_pct <= 0.5,
            rank_pct <= 0.75
        ],
        [5, 4, 3, 2],
        default=1
    )

    return plan[['Rank','Ticker','Bias','ActionScore','Entry','Target','Stop','Conviction']]


action_table, snap_date = compute_action_table(data_tidy)

latest_close = (
    data_tidy.loc[data_tidy['Date'] == snap_date, ['Ticker','Close']]
    .drop_duplicates()
)
action_table = action_table.merge(latest_close, on="Ticker", how="left")

plan = make_weekly_plan(action_table)

print(f"Weekly plan for week of {snap_date.date()}:")
print(plan.head(10).to_string(index=False))
