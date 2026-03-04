import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from backtester import PortfolioBacktest
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stock Portfolio Backtester", layout="wide")

st.title("Python Stock Backtester")
st.markdown("Replicating comprehensive quantitative analysis reports using Streamlit & yfinance.")

st.sidebar.header("Portfolio Configuration")

PREDEFINED_PORTFOLIOS = {
    "Custom": {"tickers": "AAPL, TLT, SPY", "weights": "40, 40, 20"},
    "60/40 Portfolio (Stocks/Bonds)": {"tickers": "SPY, TLT", "weights": "60, 40"},
    "All Weather (Ray Dalio)": {"tickers": "VTI, TLT, IEF, GLD, DBC", "weights": "30, 40, 15, 7.5, 7.5"},
    "Permanent Portfolio (Harry Browne)": {"tickers": "SPY, TLT, SHY, GLD", "weights": "25, 25, 25, 25"},
    "Golden Butterfly": {"tickers": "VTI, IWN, TLT, SHY, GLD", "weights": "20, 20, 20, 20, 20"}
}

st.sidebar.subheader("Strategy Selection")
selected_strategy = st.sidebar.selectbox("Choose a Strategy", list(PREDEFINED_PORTFOLIOS.keys()))

# Dynamic inputs for tickers and weights
st.sidebar.subheader("Assets")
tickers_input = st.sidebar.text_input("Tickers (comma separated)", PREDEFINED_PORTFOLIOS[selected_strategy]["tickers"])
weights_input = st.sidebar.text_input("Weights (comma separated)", PREDEFINED_PORTFOLIOS[selected_strategy]["weights"])

tickers = [t.strip() for t in tickers_input.split(",")]
try:
    weights = [float(w.strip())/100.0 for w in weights_input.split(",")]
except ValueError:
    st.sidebar.error("Weights must be numbers")
    weights = []

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime('2018-01-01'), min_value=pd.to_datetime('1990-01-01'))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime('today'), min_value=pd.to_datetime('1990-01-01'))
initial_capital = st.sidebar.number_input("Initial Capital", min_value=1000, value=10000, step=1000)
rebalance = st.sidebar.selectbox("Rebalance Frequency", ["None", "Monthly", "Annually"])
benchmark = st.sidebar.text_input("Benchmark Ticker", "SPY")

if len(tickers) != len(weights):
    st.sidebar.error("Number of tickers and weights must match.")
elif not np.isclose(sum(weights), 1.0):
    st.sidebar.error(f"Weights must sum to 100%. Currently sum to {sum(weights)*100:.1f}%")
else:
    if st.sidebar.button("Run Backtest"):
        tickers_weights = dict(zip(tickers, weights))
        
        with st.spinner("Fetching data and running backtest..."):
            bt = PortfolioBacktest(tickers_weights, start_date, end_date, initial_capital, benchmark, rebalance)
            try:
                bt.run()
                success = True
            except Exception as e:
                st.error(f"An error occurred: {e}")
                success = False
                
        if success:
            st.success("Backtest complete!")
            
            # --- Overview & Allocation ---
            st.header("1. Portfolio Overview & Allocation")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Investment Parameters**")
                st.write(f"- **Initial Balance:** ${initial_capital:,.2f}")
                st.write(f"- **Final Balance:** ${bt.portfolio_value.iloc[-1]:,.2f}")
                st.write(f"- **Rebalancing:** {rebalance}")
                st.write(f"- **Time Period:** {start_date} to {end_date}")
            with col2:
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.pie(weights, labels=tickers, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)
                
            # --- Performance & Returns ---
            st.header("2. Performance & Returns")
            
            # Growth Chart
            st.subheader("Portfolio Growth")
            growth_df = pd.DataFrame({
                'Portfolio': bt.portfolio_value,
                'Benchmark': bt.bench_value
            })
            st.line_chart(growth_df)
            
            # --- Risk & Return Metrics ---
            st.header("3. Risk & Return Metrics")
            metrics_df = pd.DataFrame({
                'Metric': ['CAGR', 'Daily Volatility (Ann.)', 'Sharpe Ratio', 'Max Drawdown', 'Beta', 'Alpha'],
                'Portfolio': [f"{bt.cagr:.2%}", f"{bt.volatility:.2%}", f"{bt.sharpe:.2f}", f"{bt.max_drawdown:.2%}", f"{bt.beta:.2f}", f"{bt.alpha:.2%}"],
                'Benchmark': [f"{bt.bench_cagr:.2%}", f"{bt.bench_volatility:.2%}", f"{bt.bench_sharpe:.2f}", f"{bt.bench_max_drawdown:.2%}", "1.00", "0.00%"]
            })
            st.table(metrics_df.set_index('Metric'))
            
            # --- Drawdowns Analysis ---
            st.header("4. Drawdowns Analysis")
            port_cummax = bt.portfolio_value.cummax()
            port_drawdown = (bt.portfolio_value / port_cummax - 1.0) * 100
            
            bench_cummax = bt.bench_value.cummax()
            bench_drawdown = (bt.bench_value / bench_cummax - 1.0) * 100
            
            dd_df = pd.DataFrame({
                'Portfolio Drawdown (%)': port_drawdown,
                'Benchmark Drawdown (%)': bench_drawdown
            })
            st.line_chart(dd_df)
            
            # --- Asset Level ---
            st.header("5. Asset-Level Analysis")
            st.subheader("Asset Correlations (Daily Returns)")
            returns = bt.prices.pct_change().dropna()
            corr = returns.corr()
            
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
            st.pyplot(fig)
