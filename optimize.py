import sys
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
from datetime import datetime
from backtester import PortfolioBacktest
import warnings

warnings.filterwarnings('ignore')

# 1. Define "Major ETFs" (No penny stock/low volume)
# We need assets with history back to ~1990 if possible, but many ETFs didn't exist then.
# To approximate long history, we might use established ones.
# SPY (1993), QQQ (1999) - Tech/Growth, TLT (2002) - Long Bonds,
# GLD (2004) - Gold,  VTI (2001) - Total Stock Market
# For testing 1990s+, we'll use mutually overlapping dates or use mutual fund proxies if needed.
# Since yfinance provides what the ETF has, we'll optimize over the max overlapping period.

TICKERS = ['SPY', 'QQQ', 'TLT', 'GLD', 'VTI', 'IEF']

def get_data():
    print("Downloading data...")
    prices = yf.download(TICKERS, start='1990-01-01', end=datetime.today().strftime('%Y-%m-%d'), threads=False)['Adj Close']
    prices.dropna(inplace=True) 
    print(f"Data available from {prices.index[0].date()} to {prices.index[-1].date()}")
    return prices


def evaluate_portfolio(weights, returns, prices, rebalance='Monthly'):
    # Run the backtester engine logic without instantiating the full class to speed up optimization,
    # or just use the class
    
    # We'll use a fast vectorized approximation for optimization: Buy & Hold or Monthly Rebalance
    # To strictly emulate Monthly Rebalance fast:
    monthly_prices = prices.resample('M').last()
    monthly_returns = monthly_prices.pct_change().dropna()
    
    port_returns = (monthly_returns * weights).sum(axis=1)
    cum_returns = (1 + port_returns).cumprod()
    
    # Metrics
    years = len(monthly_returns) / 12
    cagr = cum_returns.iloc[-1] ** (1 / years) - 1
    
    roll_max = cum_returns.cummax()
    drawdowns = cum_returns / roll_max - 1.0
    max_dd = drawdowns.min()
    
    return cagr, max_dd

def find_optimal_portfolio(prices, num_portfolios=10000):
    num_assets = len(TICKERS)
    best_weights = None
    best_score = float('inf')
    best_cagr = 0
    best_dd = 0
    
    # Generate random portfolios
    np.random.seed(42)
    weights_array = np.random.random((num_portfolios, num_assets))
    weights_array = weights_array / np.sum(weights_array, axis=1)[:, np.newaxis]
    
    monthly_prices = prices.resample('M').last()
    monthly_returns = monthly_prices.pct_change().dropna()
    
    print(f"Simulating {num_portfolios} random portfolios...")
    
    # Vectorized evaluation for speed
    # weights_array is (N, Assets)
    # monthly_returns is (Months, Assets)
    
    # (Months, N) = (Months, Assets) @ (Assets, N)
    port_returns = monthly_returns.values @ weights_array.T
    cum_returns = np.cumprod(1 + port_returns, axis=0)
    
    years = len(monthly_returns) / 12
    cagrs = cum_returns[-1, :] ** (1 / years) - 1
    
    roll_max = np.maximum.accumulate(cum_returns, axis=0)
    drawdowns = cum_returns / roll_max - 1.0
    max_dds = np.min(drawdowns, axis=0)
    
    for i in range(num_portfolios):
        cagr = cagrs[i]
        max_dd = max_dds[i]
        
        penalty = 0
        if max_dd < -0.15:
            penalty += (abs(max_dd) - 0.15) * 1000
        if cagr < 0.10:
            penalty += (0.10 - cagr) * 1000
            
        score = -cagr + penalty
        
        if score < best_score:
            best_score = score
            best_weights = weights_array[i]
            best_cagr = cagr
            best_dd = max_dd
            
    return best_weights, best_cagr, best_dd

if __name__ == "__main__":
    prices = get_data()
    
    print("Running optimization (This constraints CAGR > 10% and Max DD < 15%)...")
    optimal_weights, cagr, max_dd = find_optimal_portfolio(prices, 50000)
    
    print("\n=== Optimal Portfolio ===")
    print(f"CAGR: {cagr*100:.2f}%")
    print(f"Max Drawdown: {max_dd*100:.2f}%")
    
    print("\nWeights:")
    for t, w in zip(TICKERS, optimal_weights):
        if w > 0.01:
            print(f"  {t}: {w*100:.1f}%")

