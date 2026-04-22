import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats
import warnings

warnings.filterwarnings('ignore')

RISK_FREE_RATE = 0.04  # Assuming 4% risk free rate for simplicity

INFLATION_BENCHMARK_KEYS = {"CPI", "INFLATION", "INFLATION (CPI)"}
FRED_CPI_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"


def _fetch_cpi_series(start, end):
    """Fetch monthly CPI from FRED and return as a monthly series indexed by date."""
    df = pd.read_csv(FRED_CPI_URL)
    # FRED has used both 'DATE' and 'observation_date' as the date column name.
    date_col = next((c for c in df.columns if c.lower() in ('date', 'observation_date')), df.columns[0])
    value_col = next(c for c in df.columns if c != date_col)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: 'Date', value_col: 'CPI'}).set_index('Date')
    df['CPI'] = pd.to_numeric(df['CPI'], errors='coerce')
    df = df.dropna()
    buffer_start = pd.to_datetime(start) - pd.Timedelta(days=45)
    return df.loc[df.index >= buffer_start, 'CPI']


def _is_inflation_benchmark(name):
    if not isinstance(name, str):
        return False
    return name.strip().upper() in INFLATION_BENCHMARK_KEYS

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
        self.benchmark_is_inflation = _is_inflation_benchmark(self.benchmark)
        download_tickers = self.tickers if self.benchmark_is_inflation else self.tickers + [self.benchmark]

        data = yf.download(download_tickers, start=self.start_date, end=self.end_date, progress=False)

        # yfinance returns hierarchical columns if multiple tickers
        if 'Adj Close' in data.columns.levels[0] if isinstance(data.columns, pd.MultiIndex) else False:
            adj_close = data['Adj Close']
        else:
            # If yfinance structure changes or it's just 'Close'
            if 'Close' in data.columns.levels[0] if isinstance(data.columns, pd.MultiIndex) else False:
                 adj_close = data['Close']
            else:
                 adj_close = data

        # If it's a single ticker, yfinance may return a Series
        if isinstance(adj_close, pd.Series):
             adj_close = pd.DataFrame(adj_close, columns=[download_tickers[0]])

        adj_close = adj_close.dropna()
        self.prices = adj_close[self.tickers]

        if self.benchmark_is_inflation:
            # CPI is monthly; align to trading-day index via forward-fill.
            cpi = _fetch_cpi_series(self.start_date, self.end_date)
            self.bench_prices = cpi.reindex(self.prices.index.union(cpi.index)).sort_index().ffill()
            self.bench_prices = self.bench_prices.reindex(self.prices.index).dropna()
            # Trim prices to dates where CPI is available.
            self.prices = self.prices.loc[self.bench_prices.index]
            self.benchmark_label = "Inflation (CPI)"
        else:
            self.bench_prices = adj_close[self.benchmark]
            self.benchmark_label = self.benchmark

        if not self.prices.empty:
            actual_start_pd = self.prices.index.min()
            requested_start_pd = pd.to_datetime(self.start_date)
            if (actual_start_pd - requested_start_pd).days > 7:
                self.start_date_adjusted = True
                self.actual_start_date = actual_start_pd.strftime('%Y-%m-%d')
            else:
                self.start_date_adjusted = False
        else:
            self.start_date_adjusted = False
        
    def run(self):
        self.fetch_data()
        
        returns = self.prices.pct_change().dropna()
        self.bench_returns = self.bench_prices.pct_change().dropna()
        
        returns, self.bench_returns = returns.align(self.bench_returns, join='inner', axis=0)
        
        asset_dollars = self.initial_capital * self.weights
        bench_dollar_value = self.initial_capital
        invested_capital = self.initial_capital
        
        portfolio_dollar_values = []
        bench_values = []
        invested_capitals = []
        
        daily_returns_dollar = []
        bench_returns_dollar = []
        
        prev_portfolio_value = self.initial_capital
        prev_bench_value = self.initial_capital
        
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
            
            # 1. Market moves
            asset_dollars = asset_dollars * (1 + daily_ret.values)
            bench_dollar_value = bench_dollar_value * (1 + self.bench_returns.iloc[i])
            
            curr_portfolio_value = np.sum(asset_dollars)
            
            # 2. Daily returns based on dollars (before any cash flow for the day)
            port_ret = (curr_portfolio_value / prev_portfolio_value) - 1.0 if prev_portfolio_value > 0 else 0
            bench_ret = (bench_dollar_value / prev_bench_value) - 1.0 if prev_bench_value > 0 else 0
            
            daily_returns_dollar.append(port_ret)
            bench_returns_dollar.append(bench_ret)
            
            # 3. Actions (Rebalance & Installment applied End-of-Day)
            if is_rebalance:
                asset_dollars = curr_portfolio_value * self.weights
                
            if is_installment:
                asset_dollars += self.installment_amount * self.weights
                bench_dollar_value += self.installment_amount
                invested_capital += self.installment_amount
                curr_portfolio_value += self.installment_amount
            
            portfolio_dollar_values.append(curr_portfolio_value)
            bench_values.append(bench_dollar_value)
            invested_capitals.append(invested_capital)
            
            prev_portfolio_value = curr_portfolio_value
            prev_bench_value = bench_dollar_value
            
        self.portfolio_returns = pd.Series(daily_returns_dollar, index=returns.index)
        self.cumulative_returns = (1 + self.portfolio_returns).cumprod()
        self.portfolio_value = pd.Series(portfolio_dollar_values, index=returns.index)
        
        self.bench_cumulative = (1 + pd.Series(bench_returns_dollar, index=returns.index)).cumprod()
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

    def rolling_start_analysis(self, min_window_days=60):
        """For each candidate start date (1st trading day of each month), compute
        the CAGR and total return that would have resulted from holding through
        to self.end_date, using the already-simulated daily strategy returns.

        Note: this treats each start as a lump-sum investment under the same
        rebalance policy. Installment cash flows are not re-simulated; the
        daily returns series already excludes cash-flow effects, so this is
        exact for lump-sum and a close approximation for DCA.
        """
        port_rets = self.portfolio_returns
        bench_rets = self.bench_returns.reindex(port_rets.index).fillna(0)

        if port_rets.empty:
            return pd.DataFrame(columns=[
                'Portfolio CAGR', 'Benchmark CAGR',
                'Portfolio Total Return', 'Benchmark Total Return',
            ])

        # Use the first trading day of each month as a start candidate.
        monthly_firsts = port_rets.groupby([port_rets.index.year, port_rets.index.month]).apply(
            lambda s: s.index.min()
        ).tolist()

        records = []
        for start in monthly_firsts:
            window = port_rets.loc[start:]
            bwindow = bench_rets.loc[start:]
            if len(window) < min_window_days:
                continue

            days = len(window)
            years = days / 252.0

            p_cum = float((1 + window).prod())
            b_cum = float((1 + bwindow).prod())

            p_cagr = p_cum ** (1 / years) - 1 if years > 0 else 0.0
            b_cagr = b_cum ** (1 / years) - 1 if years > 0 else 0.0

            records.append({
                'start_date': start,
                'Portfolio CAGR': p_cagr,
                'Benchmark CAGR': b_cagr,
                'Portfolio Total Return': p_cum - 1.0,
                'Benchmark Total Return': b_cum - 1.0,
                'Years Held': years,
            })

        return pd.DataFrame(records).set_index('start_date')
