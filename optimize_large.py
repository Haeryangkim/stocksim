import sys
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Extensive list of major ETF asset classes:
# SPY/VTI: US Equity
# QQQ: Tech Growth
# DVY: US High Dividend (Proxy for SCHD to allow 2004 start date, since SCHD launched 2011)
# EFA: Developed Markets
# EEM: Emerging Markets
# VNQ: Real Estate
# TLT: Long-term Treasuries
# IEF: Mid-term Treasuries
# SHY: Short-term Treasuries
# GLD: Gold (Launched Nov 18, 2004)
TICKERS = ['SPY', 'QQQ', 'VTI', 'DVY', 'EFA', 'EEM', 'VNQ', 'TLT', 'IEF', 'SHY', 'GLD']

def get_data():
    print("Loading data from etf_prices_large.csv...")
    prices = pd.read_csv("etf_prices_large.csv", index_col=0, parse_dates=True)
    prices.index = pd.to_datetime(prices.index, utc=True)
    print(f"Data available from {prices.index[0].date()} to {prices.index[-1].date()}")
    return prices

def evaluate_portfolio(weights, monthly_returns):
    port_returns = monthly_returns.values @ weights
    cum_returns = np.cumprod(1 + port_returns)
    
    years = len(monthly_returns) / 12
    cagr = cum_returns[-1] ** (1 / years) - 1
    
    roll_max = np.maximum.accumulate(cum_returns)
    drawdowns = cum_returns / roll_max - 1.0
    max_dd = np.min(drawdowns)
    
    return cagr, max_dd

def objective(weights, monthly_returns):
    cagr, max_dd = evaluate_portfolio(weights, monthly_returns)
    
    # Penalties
    penalty = 0
    if max_dd < -0.15:
        penalty += (abs(max_dd) - 0.15) * 1000  # heavy penalty for DD > 15%
        
    if cagr < 0.10:
        penalty += (0.10 - cagr) * 1000  # penalty for CAGR < 10%
        
    # We want to maximize CAGR and minimize Max DD, so we minimize negative CAGR + penalty
    score = -cagr + penalty
    return score

def find_optimal_portfolio(prices):
    num_assets = len(TICKERS)
    
    monthly_prices = prices.resample('M').last()
    monthly_returns = monthly_prices.pct_change().dropna()
    
    # Random search approach (Differential Evolution) for global minimum
    from scipy.optimize import differential_evolution
    
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    
    # We need to ensure weights sum to 1. Differential evolution natively handles bounds but not sum constraints well.
    # We'll normalize weights inside the objective wrapper.
    def obj_normalized(weights):
        w = weights / np.sum(weights)
        return objective(w, monthly_returns)
    
    print("Running Differential Evolution Optimization (Focus: >10% CAGR, <15% DD)...")
    result = differential_evolution(obj_normalized, bounds, maxiter=200, popsize=30, tol=1e-4, seed=42)
    
    optimal_weights = result.x / np.sum(result.x)
    cagr, max_dd = evaluate_portfolio(optimal_weights, monthly_returns)
    
    print("\n=== Optimal Portfolio (2004 - Present) ===")
    print(f"CAGR: {cagr*100:.2f}%")
    print(f"Max Drawdown: {max_dd*100:.2f}%")
    
    print("\nWeights:")
    for t, w in zip(TICKERS, optimal_weights):
        # Round up to 1 decimal place and ignore anything less than ~0.5%
        if w >= 0.005:
            print(f"  {t}: {w*100:.1f}%")

if __name__ == "__main__":
    prices = get_data()
    find_optimal_portfolio(prices)
