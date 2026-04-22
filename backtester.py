import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats
import warnings

warnings.filterwarnings('ignore')

RISK_FREE_RATE = 0.04  # Assuming 4% risk free rate for simplicity

class PortfolioBacktest:
    def __init__(self, tickers_weights, start_date, end_date, initial_capital=10000, benchmark='SPY', rebalance='Monthly', installment_amount=0, installment_frequency='None'):
        self.tickers_weights = tickers_weights
        self.tickers = list(tickers_weights.keys())
        self.weights = np.array(list(tickers_weights.values()))
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.benchmark = benchmark
        self.rebalance = rebalance
        self.installment_amount = installment_amount
        self.installment_frequency = installment_frequency
        
        if not np.isclose(sum(self.weights), 1.0):
            raise ValueError("Weights must sum to 1.0")
            
        self.prices = pd.DataFrame()
        self.bench_prices = pd.Series(dtype=float)
        
    def fetch_data(self):
        all_tickers = self.tickers + [self.benchmark]
        # Download data
        data = yf.download(all_tickers, start=self.start_date, end=self.end_date, progress=False)
        
        # yfinance returns hierarchical columns if multiple tickers
        if 'Adj Close' in data.columns.levels[0] if isinstance(data.columns, pd.MultiIndex) else False:
            adj_close = data['Adj Close']
        else:
            # If yfinance structure changes or it's just 'Close'
            if 'Close' in data.columns.levels[0] if isinstance(data.columns, pd.MultiIndex) else False:
                 adj_close = data['Close']
            else:
                 adj_close = data
                 
        # If it's single ticker + benchmark, it might just be two columns under Adj Close
        if isinstance(adj_close, pd.Series):
             adj_close = pd.DataFrame(adj_close, columns=[all_tickers[0]])
             
        adj_close = adj_close.dropna()
        self.prices = adj_close[self.tickers]
        self.bench_prices = adj_close[self.benchmark]
        
    def run(self):
        self.fetch_data()
        
        returns = self.prices.pct_change().dropna()
        self.bench_returns = self.bench_prices.pct_change().dropna()
        
        returns, self.bench_returns = returns.align(self.bench_returns, join='inner', axis=0)
        
        daily_returns = []
        current_weights = self.weights.copy()
        
        current_dollar_value = self.initial_capital
        bench_dollar_value = self.initial_capital
        invested_capital = self.initial_capital
        
        portfolio_dollar_values = []
        bench_values = []
        invested_capitals = []
        
        dates = returns.index
        for i, (date, daily_ret) in enumerate(returns.iterrows()):
            is_rebalance = False
            is_installment = False
            
            if i > 0:
                prev_date = dates[i-1]
                month_changed = date.month != prev_date.month
                year_changed = date.year != prev_date.year
                
                if self.rebalance == 'Monthly' and month_changed:
                    is_rebalance = True
                elif self.rebalance == 'Annually' and year_changed:
                    is_rebalance = True
                
                if self.installment_frequency == 'Monthly' and month_changed:
                    is_installment = True
                elif self.installment_frequency == 'Annually' and year_changed:
                    is_installment = True
            
            if is_rebalance:
                current_weights = self.weights.copy()
            
            port_ret = np.dot(current_weights, daily_ret.values)
            daily_returns.append(port_ret)
            
            # Compound the dollar value BEFORE the end-of-period installment is added
            current_dollar_value = current_dollar_value * (1 + port_ret)
            bench_dollar_value = bench_dollar_value * (1 + self.bench_returns.iloc[i])
            
            if is_installment:
                current_dollar_value += self.installment_amount
                bench_dollar_value += self.installment_amount
                invested_capital += self.installment_amount
            
            portfolio_dollar_values.append(current_dollar_value)
            bench_values.append(bench_dollar_value)
            invested_capitals.append(invested_capital)
            
            current_weights = current_weights * (1 + daily_ret.values)
            sum_weights = np.sum(current_weights)
            if sum_weights > 0:
                current_weights = current_weights / sum_weights
            
        self.portfolio_returns = pd.Series(daily_returns, index=returns.index)
        self.cumulative_returns = (1 + self.portfolio_returns).cumprod()
        self.portfolio_value = pd.Series(portfolio_dollar_values, index=returns.index)
        
        self.bench_cumulative = (1 + self.bench_returns).cumprod()
        self.bench_value = pd.Series(bench_values, index=returns.index)
        self.invested_capitals = pd.Series(invested_capitals, index=returns.index)
        
        self.calculate_metrics()
        
    def calculate_metrics(self):
        days = len(self.portfolio_returns)
        years = days / 252.0 if days > 0 else 1
        
        # CAGR
        self.cagr = self.cumulative_returns.iloc[-1] ** (1 / years) - 1
        self.bench_cagr = self.bench_cumulative.iloc[-1] ** (1 / years) - 1
        
        # Volatility
        self.volatility = self.portfolio_returns.std() * np.sqrt(252)
        self.bench_volatility = self.bench_returns.std() * np.sqrt(252)
        
        # Sharpe
        self.sharpe = (self.cagr - RISK_FREE_RATE) / self.volatility if self.volatility > 0 else 0
        self.bench_sharpe = (self.bench_cagr - RISK_FREE_RATE) / self.bench_volatility if self.bench_volatility > 0 else 0
        
        # Max Drawdown
        roll_max = self.cumulative_returns.cummax()
        drawdowns = self.cumulative_returns / roll_max - 1.0
        self.max_drawdown = drawdowns.min()
        
        bench_roll_max = self.bench_cumulative.cummax()
        bench_drawdowns = self.bench_cumulative / bench_roll_max - 1.0
        self.bench_max_drawdown = bench_drawdowns.min()
        
        # Beta & Alpha
        cov = np.cov(self.portfolio_returns, self.bench_returns)[0][1]
        var = np.var(self.bench_returns)
        self.beta = cov / var if var > 0 else 1
        self.alpha = self.cagr - (RISK_FREE_RATE + self.beta * (self.bench_cagr - RISK_FREE_RATE))
