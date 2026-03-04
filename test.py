import sys
sys.path.append('.')
from backtester import PortfolioBacktest

if __name__ == "__main__":
    print("Testing backtester...")
    bt = PortfolioBacktest({'AAPL': 0.6, 'TLT': 0.4}, '2020-01-01', '2023-01-01', rebalance='Monthly')
    bt.run()
    
    print(f"CAGR: {bt.cagr:.2%}")
    print(f"Volatility: {bt.volatility:.2%}")
    print(f"Sharpe: {bt.sharpe:.2f}")
    print(f"Max DD: {bt.max_drawdown:.2%}")
    print(f"Beta: {bt.beta:.2f}")
    print(f"Alpha: {bt.alpha:.2%}")
    
    print("Test passed!")
