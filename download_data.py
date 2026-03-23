import yfinance as yf
from datetime import datetime
import pandas as pd

TICKERS = ['SPY', 'QQQ', 'VTI', 'DVY', 'EFA', 'EEM', 'VNQ', 'TLT', 'IEF', 'SHY', 'GLD','SCHD']
print("Downloading data one by one...")
all_data = {}
for t in TICKERS:
    print(f"Fetching {t}...")
    try:
        # get historical market data
        ticker_obj = yf.Ticker(t)
        hist = ticker_obj.history(period="max")
        if not hist.empty:
            all_data[t] = hist['Close']
            print(f"  -> Got data from {hist.index[0].date()} to {hist.index[-1].date()}")
        else:
            print(f"  -> No data found for {t}")
    except Exception as e:
        print(f"  -> Error fetching {t}: {e}")

if all_data:
    df = pd.DataFrame(all_data)
    df = df[df.index >= pd.to_datetime('2004-11-18').tz_localize(df.index.tz)]
    df.dropna(inplace=True)
    df.to_csv("etf_prices_large.csv")
    print(f"\nSaved combined data to etf_prices_large.csv: {df.shape[0]} rows covering dates from {df.index[0].date()} to {df.index[-1].date()}")
